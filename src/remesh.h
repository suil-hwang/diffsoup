// src/remesh.h
// Adaptive triangle-soup subdivision in world space.

#pragma once

#include <vector>
#include <queue>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace diffsoup {

class TriangleSoupSplitter {
private:
    static constexpr int vertexProvenanceWidth = 3;

    std::vector<float> vertices;   // flat xyz
    std::vector<int>   triangles;  // flat i0,i1,i2
    int originalNumTriangles;
    bool trackVertexProvenance;

    // Direct affine recipe over input vertices, three entries per output.
    std::vector<int>   vertexSourceIndices;
    std::vector<float> vertexSourceWeights;

    // current-face → original-face id it descends from
    std::vector<int> faceMapping;

    // 1 = current triangle is exactly the original (never modified), 0 otherwise
    std::vector<unsigned char> sameAsOriginal;

    // Per-triangle generation counter (bumped on every edit)
    std::vector<int> triGen;

    // Per-triangle origin: which original face this triangle descends from
    std::vector<int> triOrigin;

    struct EdgeRef {
        int tri;        // triangle index
        int e;          // edge: 0:(v0→v1), 1:(v1→v2), 2:(v2→v0)
        float len2;     // squared length at the time of push
        int gen;        // triangle generation when this was computed
        bool operator<(const EdgeRef& other) const { return len2 < other.len2; }
    };

    inline int numVerts() const { return static_cast<int>(vertices.size() / 3); }
    inline const float* p(int vi) const { return &vertices[3 * vi]; }

    inline float lengthSquaredBetween(int a, int b) const {
        const float* A = p(a);
        const float* B = p(b);
        float dx = B[0] - A[0];
        float dy = B[1] - A[1];
        float dz = B[2] - A[2];
        return dx*dx + dy*dy + dz*dz;
    }

    inline void triIndices(int t, int& i0, int& i1, int& i2) const {
        const int base = 3 * t;
        i0 = triangles[base + 0];
        i1 = triangles[base + 1];
        i2 = triangles[base + 2];
    }

    inline void setTri(int t, int i0, int i1, int i2) {
        const int base = 3 * t;
        triangles[base + 0] = i0;
        triangles[base + 1] = i1;
        triangles[base + 2] = i2;
    }

    inline int addVertex(
        float x, float y, float z,
        int source0, int source1,
        float weight0, float weight1);
    inline int copyVertex(int src);
    inline void appendBlendedVertexProvenance(
        int source0, int source1,
        float weight0, float weight1);

    void enqueueTriangleEdges(int t, std::priority_queue<EdgeRef>& pq) const;
    int splitTriangleEdge(int t, int e);

public:
    TriangleSoupSplitter(
        const float* verts,
        const int* tris,
        int nv,
        int nt,
        bool track_vertex_provenance = false);

    void splitLongEdges(int numSplits) { splitLongEdges(numSplits, 0.0f); }
    void splitLongEdges(int numSplits, float tau);
    void splitLongEdgesUntil(float tau, int hardCap = -1) { splitLongEdges(hardCap, tau); }

    int getNumVertices() const;
    int getNumTriangles() const;
    int getOriginalNumTriangles() const;

    void exportToFlatArrays(float* outVerts, int* outFaces) const;
    void getFaceMapping(int* outMapping) const;
    void getSameAsOriginal(int* outFlags) const;
    void getVertexProvenance(int* outIndices, float* outWeights) const;
};

} // namespace diffsoup
