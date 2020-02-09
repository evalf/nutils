// This geo file defines the [0,2]x[0,1] domain required by test_mesh.gmsh,
// with "left" and "right" volume groups, "neumann" and "dirichlet" boundary
// groups, an "iface" interface group separating "left" and "right", and a
// "midpoint" point group at coordinate (1,0). To regenerate the msh files:
//
// for o in 1 2 3 4; do for v in 2 4; do gmsh -format msh$v -2 -order $o mesh2d.geo -o mesh2d_p${o}_v${v}.msh; done; done

p00 = newp; Point(p00) = {0,0,0};
p01 = newp; Point(p01) = {0,1,0};
p10 = newp; Point(p10) = {1,0,0};
p11 = newp; Point(p11) = {1,1,0};
p20 = newp; Point(p20) = {2,0,0};
p21 = newp; Point(p21) = {2,1,0};
l0X = newl; Line(l0X) = {p00,p01};
l1X = newl; Line(l1X) = {p10,p11};
l2X = newl; Line(l2X) = {p20,p21};
lL0 = newl; Line(lL0) = {p00,p10};
lL1 = newl; Line(lL1) = {p01,p11};
lR0 = newl; Line(lR0) = {p10,p20};
lR1 = newl; Line(lR1) = {p11,p21};
llL = newll; Line Loop(llL) = {lL0,l1X,-lL1,-l0X};
llR = newll; Line Loop(llR) = {lR0,l2X,-lR1,-l1X};
sL = news; Plane Surface(sL) = {llL};
sR = news; Plane Surface(sR) = {llR};
Physical Point("midpoint") = {p10};
Physical Line("neumann") = {lL0,lR0};
Physical Line("dirichlet") = {l2X,lR1,lL1,l0X};
Physical Line("iface") = {l1X};
Physical Surface("left") = {sL};
Physical Surface("right") = {sR};
