References:
- https://en.wikipedia.org/wiki/Minkowski_addition
- https://doc.cgal.org/latest/Minkowski_sum_2/
- https://egeozgul.github.io/Minkowski-Sum-Calculator/ (comparing to see if the algorithm implemented is working correctly)
- https://github.com/grzesiek2201/MinkowskiSum (python implementation)
- https://cp-algorithms.com/geometry/minkowski.html#implementation (distance between polygons using minkowski difference)
- https://codeforces.com/blog/entry/103477
- https://stackoverflow.com/questions/10983872/distance-from-a-point-to-a-polygon (distance from a point to a polygon pseudocode)
- https://en.wikipedia.org/wiki/Point_in_polygon#Ray_casting_algorithm (ray casting algorithm to check if point is inside polygon)

A complexidade do algoritmo da soma de Minkowski é O(m+n), onde m e n são o número de vértices em cada polígono. No caso da expansão de polígonos, a soma é repetida entre o polígono "robô" e os polígonos de obstáculo, tornando a complexidade O(M*N).

Vídeo do código rodando:
https://drive.google.com/file/d/1Y3lqEB9tL8tvCZ2NbrwEdIYlkdgbwWaz/view?usp=drive_link