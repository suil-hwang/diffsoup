// src/remesh_clip.h
// Adaptive triangle-soup subdivision in clip / screen space.

#pragma once

#include <vector>
#include <queue>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace diffsoup {

class TriangleSoupSplitterClip {
private:
    static constexpr int vertexProvenanceWidth = 3;

    std::vector<float> vertices;   // flat xyz
    std::vector<int>   triangles;  // flat i0,i1,i2
    std::vector<int>   valid_triangles;
    int originalNumTriangles;
    bool trackVertexProvenance;

    float mvp4x4[16];

    std::vector<int>            faceMapping;
    std::vector<unsigned char>  sameAsOriginal;
    std::vector<int>            triGen;
    std::vector<int>            triOrigin;
    // Direct affine recipe over input vertices, three entries per output.
    std::vector<int>            vertexSourceIndices;
    std::vector<float>          vertexSourceWeights;

    struct EdgeRef {
        int tri, e;
        float len2;
        int gen;
        bool operator<(const EdgeRef& o) const { return len2 < o.len2; }
    };

    inline const float* p(int vi) const { return &vertices[3 * vi]; }

    inline float screenLen2Between(int a, int b, float aspectWH) const;

    inline void triIndices(int t, int& i0, int& i1, int& i2) const;
    inline void setTri(int t, int i0, int i1, int i2);
    inline int  addVertex(
        float x, float y, float z,
        int source0, int source1,
        float weight0, float weight1);
    inline int  copyVertex(int src);
    inline void appendBlendedVertexProvenance(
        int source0, int source1,
        float weight0, float weight1);

    void enqueueTriangleEdges(int t, std::priority_queue<EdgeRef>& pq, float aspectWH) const;
    int  splitTriangleEdge(int t, int e);

public:
    TriangleSoupSplitterClip(
        const float* mvp,
        const float* verts,
        const int* tris,
        int nv, int nt,
        const int* valid_tris,
        bool track_vertex_provenance = false);

    void splitLongEdges(int numSplits) { splitLongEdges(numSplits, 0.0f, 1.0f); }
    void splitLongEdges(int numSplits, float tau_ratio, float aspectWH);
    void splitLongEdgesUntil(float tau_ratio, int hardCap = -1, float aspectWH = 1.0f) {
        splitLongEdges(hardCap, tau_ratio, aspectWH);
    }

    int  getNumVertices() const;
    int  getNumTriangles() const;
    int  getOriginalNumTriangles() const;

    void exportToFlatArrays(float* outVerts, int* outFaces) const;
    void getFaceMapping(int* outMapping) const;
    void getSameAsOriginal(int* outFlags) const;
    void getVertexProvenance(int* outIndices, float* outWeights) const;
};

} // namespace diffsoup
