from collections import defaultdict
from dataStructures import Triangle, Point

def bowyer_watson(points):
    super_triangle = [Point(-1e5, -1e5), Point(1e5, -1e5), Point(0, 1e5)]
    triangulation = [Triangle(*super_triangle)]# start the triangulation with only the super triangle

    # add each point 1 by 1
    for p in points:
        # find all triangles who's circumference contains the new point
        bad_triangles = [t for t in triangulation if t.circumcircle_contains(p)]
        if not bad_triangles:
            # skip when no bad triangles are found
            continue
        
        # edges that appear only once are the boundaries of the hole, first count edges, then filter the boundary out
        edge_count = defaultdict(int)
        for t in bad_triangles:
            vs = t.vertices
            for i in range(3):
                a = vs[i]
                b = vs[(i + 1) % 3]
                if (b, a) in edge_count:
                    edge_count[(b, a)] += 1
                else:
                    edge_count[(a, b)] += 1

        boundary = []
        for (a, b), cnt in edge_count.items():
            if cnt == 1:
                boundary.append((a, b))

        # remove bad triangles
        triangulation = [t for t in triangulation if t not in bad_triangles]

        # create new triangles
        for (a, b) in boundary:
            new_tri = Triangle(a, b, p)
            triangulation.append(new_tri)

    # after adding all points, remove any triangles that share a vertex with the super triangle
    result = []
    super_set = set(super_triangle)
    for t in triangulation:
        skip = False
        for v in t.vertices:
            if v in super_set:
                skip = True
                break
        if not skip:
            result.append(t)

    return result
