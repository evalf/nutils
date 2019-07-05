# Rectangular domain.
gmsh -format msh2 -clmin .5 -2 rectangle.geo -o rectangle.msh

# Square periodic domain. Note that clmin<=.4 is required because we need at
# least 4 nodes on the non-periodic boundaries.
gmsh -format msh2 -clmin .4 -2 periodic.geo -o periodic.msh

# Square domain with several polynomial orders
gmsh -format msh2 -clmin .5 -2 -order 1 square.geo -o square_p1.msh
gmsh -format msh2 -clmin .5 -2 -order 2 square.geo -o square_p2.msh

# Cubic domain
gmsh -format msh2 -clmin 1 -3 -order 1 cube.geo -o cube_p1.msh
