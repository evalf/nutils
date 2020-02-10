// This geo file defines the unit semi-sphere without physical groups,
// resulting in a tagless 2D topology with a 3D geometry.
//
// for o in 1 2; do for v in 2 4; do gmsh -format msh$v -3 -order $o mesh3dmani.geo -o mesh3dmani_p${o}_v${v}.msh; done; done

Point(1) = {0,0,0,.5};
Point(2) = {1,0,0,.5};
Point(3) = {0,1,0,.5};
Point(4) = {0,0,1,.5};
Point(5) = {-1,0,0,.5};
Point(6) = {0,-1,0,.5};

Circle(1) = {2,1,3};
Circle(2) = {3,1,5};
Circle(3) = {5,1,6};
Circle(4) = {6,1,2};
Circle(5) = {5,1,4};
Circle(6) = {4,1,2};
Circle(7) = {3,1,4};
Circle(8) = {4,1,6};

Line Loop(1) = {1,7,6};
Line Loop(2) = {2,5,-7};
Line Loop(3) = {3,-8,-5};
Line Loop(4) = {4,-6,8};

Surface(1) = {1};
Surface(2) = {2};
Surface(3) = {3};
Surface(4) = {4};
