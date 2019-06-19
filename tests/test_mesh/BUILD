# Rectangular domain.
gmsh -format msh2 -clmin .5 -2 rectangle.geo -o rectangle.msh

# Square periodic domain. Note that clmin<=.4 is required because we need at
# least 4 nodes on the non-periodic boundaries.
gmsh -format msh2 -clmin .4 -2 periodic.geo -o periodic.msh

# Square domain with several polynomial orders
gmsh -format msh2 -clmin .5 -2 -order 1 square.geo -o square_p1.msh
gmsh -format msh2 -clmin .5 -2 -order 2 square.geo -o square_p2.msh
gmsh -format msh2 -clmin .5 -2 -order 3 square.geo -o square_p3.msh
gmsh -format msh2 -clmin .5 -2 -order 4 square.geo -o square_p4.msh

# Cubic domain with several polynomial orders
gmsh -format msh2 -clmin 1 -3 -order 1 cube.geo -o cube_p1.msh
gmsh -format msh2 -clmin 1 -3 -order 2 cube.geo -o cube_p2.msh
