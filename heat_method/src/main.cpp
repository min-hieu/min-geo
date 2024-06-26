#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/vertex_position_geometry.h"
#include "geometrycentral/surface/heat_method_distance.h"
#include "geometrycentral/surface/poisson_disk_sampler.h"

#include "geometrycentral/pointcloud/point_cloud.h"
#include "geometrycentral/pointcloud/point_position_geometry.h"

#include "geometrycentral/numerical/linear_solvers.h"

#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"

#include "args/args.hxx"
#include "imgui.h"

using namespace geometrycentral;
using namespace geometrycentral::surface;
using namespace geometrycentral::pointcloud;

// == Geometry-central data
std::unique_ptr<ManifoldSurfaceMesh> mesh;
std::unique_ptr<VertexPositionGeometry> meshGeom;
std::unique_ptr<PointCloud> pc;
std::unique_ptr<PointPositionGeometry> pcGeom;

// Polyscope visualization handle, to quickly add data to the surface
polyscope::SurfaceMesh *psMesh;

// Some algorithm parameters
float param1 = 42.0;

// Example computation function -- this one computes and registers a scalar
// quantity
void computeHeatGeodesicReimplementation() {
  double tCoef = 1.0;

  meshGeom->requireEdgeLengths();
  meshGeom->requireVertexIndices();
  meshGeom->requireFaceIndices();
  meshGeom->requireFaceTangentBasis();
  meshGeom->requireVertexLumpedMassMatrix();
  meshGeom->requireCotanLaplacian();
  meshGeom->requireHalfedgeCotanWeights();
  meshGeom->requireHalfedgeVectorsInFace();

  double meanEdgeLength = 0.;
  for (Edge e : mesh->edges()) {
    meanEdgeLength += meshGeom->edgeLengths[e];
  }
  meanEdgeLength /= mesh->nEdges();
  double t = tCoef * meanEdgeLength * meanEdgeLength;
  SparseMatrix<double> M = meshGeom->vertexLumpedMassMatrix;
  SparseMatrix<double> L = meshGeom->cotanLaplacian;
  SparseMatrix<double> HeatOp = M + t*L;

  Vertex pointSrc = mesh->vertex(1162);
  VertexData<double> heatRHS(*mesh, 0.);
  heatRHS[pointSrc] = 1.;

  Vector<double> u = solvePositiveDefinite(HeatOp, heatRHS.toVector());
  Vector<double> divX = Vector<double>::Zero(mesh->nVertices());
  FaceData<Vector2> grad_u_face(*mesh);
  FaceData<Vector3> fBasisX(*mesh);
  FaceData<Vector3> fBasisY(*mesh);

  for (Face f: mesh->faces()) {

    Vector2 grad_u = Vector2::zero();
    for (Halfedge he: f.adjacentHalfedges()) {
      meshGeom->halfedgeVectorsInFace[he.next()];
      Vector2 perp_e = meshGeom->halfedgeVectorsInFace[he.next()].rotate90();
      grad_u += u(meshGeom->vertexIndices[he.vertex()]) * perp_e;
    }

    Vector2 X = grad_u.normalizeCutoff();

    grad_u_face[f] = X;
    fBasisX[meshGeom->faceIndices[f]] = meshGeom->faceTangentBasis[f][0];
    fBasisY[meshGeom->faceIndices[f]] = meshGeom->faceTangentBasis[f][1];

    for (Halfedge he: f.adjacentHalfedges()) {
      Vector2 e2 = meshGeom->halfedgeVectorsInFace[he];
      Vector2 e1 = meshGeom->halfedgeVectorsInFace[he.next().next()];
      double val = meshGeom->halfedgeCotanWeights[he] * dot(e2,X)
                   - meshGeom->halfedgeCotanWeights[he.next().next()] * dot(e1,X);
      divX(meshGeom->vertexIndices[he.vertex()]) += val;
    }
  }

  Vector<double> distToSource = solvePositiveDefinite(L, divX);
  distToSource = distToSource.array() - distToSource(meshGeom->vertexIndices[pointSrc]);

  psMesh->addVertexScalarQuantity("our heat distance", distToSource);
  psMesh->addFaceTangentVectorQuantity("our heat direction", grad_u_face, fBasisX, fBasisY);

  meshGeom->unrequireEdgeLengths();
  meshGeom->unrequireVertexIndices();
  meshGeom->unrequireFaceIndices();
  meshGeom->unrequireFaceTangentBasis();
  meshGeom->unrequireVertexLumpedMassMatrix();
  meshGeom->unrequireCotanLaplacian();
  meshGeom->unrequireHalfedgeCotanWeights();
  meshGeom->unrequireHalfedgeVectorsInFace();
}

void computeHeatGeodesicReference() {
  Vertex heatSource = mesh->vertex(1162);
  VertexData<double> distToSource = heatMethodDistance(*meshGeom, heatSource);
  psMesh->addVertexScalarQuantity("ref heat distance", distToSource);
}

void guiCallback() {

  if (ImGui::Button("reimplemented heat method")) {
    computeHeatGeodesicReimplementation();
  }

  if (ImGui::Button("reference heat method")) {
    computeHeatGeodesicReference();
  }

  if (ImGui::Button("get point cloud")) {
    PointData<Vector3> pointPos;
    PointData<SurfacePoint> cloudSources;
    size_t nPts = 15000;
    std::tie(pc, pointPos, cloudSources) = uniformlySamplePointsOnSurface(*mesh, *meshGeom, nPts);
  }
}

int main(int argc, char **argv) {

  // Configure the argument parser
  args::ArgumentParser parser("meshGeom-central & Polyscope example project");
  args::Positional<std::string> inputFilename(parser, "mesh", "A mesh file.");

  // Parse args
  try {
    parser.ParseCLI(argc, argv);
  } catch (args::Help &h) {
    std::cout << parser;
    return 0;
  } catch (args::ParseError &e) {
    std::cerr << e.what() << std::endl;
    std::cerr << parser;
    return 1;
  }

  // Make sure a mesh name was given
  if (!inputFilename) {
    std::cerr << "Please specify a mesh file as argument" << std::endl;
    return EXIT_FAILURE;
  }

  // Initialize polyscope
  polyscope::init();
  polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::ShadowOnly;

  // Set the callback function
  polyscope::state::userCallback = guiCallback;

  // Load mesh
  std::tie(mesh, meshGeom) = readManifoldSurfaceMesh(args::get(inputFilename));

  // Register the mesh with polyscope
  psMesh = polyscope::registerSurfaceMesh(
      polyscope::guessNiceNameFromPath(args::get(inputFilename)),
      meshGeom->inputVertexPositions, mesh->getFaceVertexList(),
      polyscopePermutations(*mesh));

  polyscope::show();

  return EXIT_SUCCESS;
}
