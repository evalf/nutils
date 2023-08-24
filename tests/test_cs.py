from nutils import mesh, function, cs
import unittest
import numpy


class TestShape(unittest.TestCase):

    def assertAreaCentroid(self, topo, geom, *, mass, centroid, places, degree=1):
        J = function.J(geom)
        _mass, _moment = topo.sample('gauss', 2).integrate([J, geom * J], degree=degree*2)
        self.assertAlmostEqual(_mass, mass, places=places)
        numpy.testing.assert_almost_equal(actual=_moment/mass, desired=centroid, decimal=places)

    def test_rectangle(self):
        'Square with 1<x<2 and 3<y<4'
        for periodic in ('', 'x', 'y', 'xy'):
            rect = cs.Rectangle(cs.Interval(1, 2, periodic='x' in periodic), cs.Interval(3, 4, periodic='y' in periodic))
            shapes = dict(dom=rect,
                left=rect.boundary['left'],
                right=rect.boundary['right'],
                bottom=rect.boundary['bottom'],
                top=rect.boundary['top'])
            topo, geom = mesh.csgmsh(shapes, elemsize=.1)
            with self.subTest(f'interior(periodic={periodic})'):
                self.assertAreaCentroid(topo, geom, mass=1, centroid=(1.5,3.5), places=10)
            volume = 0
            if 'x' not in periodic:
                with self.subTest(f'left,right(periodic={periodic})'):
                    self.assertAreaCentroid(topo.boundary['left'], geom, mass=1, centroid=(1,3.5), places=10)
                    self.assertAreaCentroid(topo.boundary['right'], geom, mass=1, centroid=(2,3.5), places=10)
                volume += 2
            if 'y' not in periodic:
                with self.subTest(f'bottom,top(periodic={periodic})'):
                    self.assertAreaCentroid(topo.boundary['bottom'], geom, mass=1, centroid=(1.5,3), places=10)
                    self.assertAreaCentroid(topo.boundary['top'], geom, mass=1, centroid=(1.5,4), places=10)
                volume += 2
            with self.subTest(f'boundary(periodic={periodic})'):
                if volume:
                    self.assertAreaCentroid(topo.boundary, geom, mass=volume, centroid=(1.5,3.5), places=10)
                else:
                    self.assertEqual(len(topo.boundary), 0)

    def test_circle(self):
        'Circle with radius .5 centered around x=1 y=2'
        circ = cs.Circle(center=(1.,2.),radius=.5)
        topo, geom = mesh.csgmsh(dict(dom=circ), elemsize=.1, order=2)
        with self.subTest('interior'):
            self.assertAreaCentroid(topo, geom, mass=.25*numpy.pi, centroid=(1,2), degree=2, places=5)
        with self.subTest('boundary'):
            self.assertAreaCentroid(topo.boundary, geom, mass=numpy.pi, centroid=(1,2), degree=2, places=5)

    def test_ellipse(self):
        'Ellipse with width .5, height 1, centered around x=1 y=2 and rotated by 30deg'
        ellipse = cs.Ellipse(center=(1.,2.),width=1.,height=2.,angle=30.)
        topo, geom = mesh.csgmsh(dict(dom=ellipse), elemsize=.1, order=2)
        with self.subTest('interior'):
            self.assertAreaCentroid(topo, geom, mass=.5*numpy.pi, centroid=(1,2), degree=2, places=5)
        with self.subTest('boundary'):
            self.assertAreaCentroid(topo.boundary, geom, mass=4.84422411, centroid=(1,2), degree=2, places=4)

    def test_path(self):
        'Quarter wedge with radius 2 centered around x=3 y=2'
        path = cs.Path(vertices=((1,2),(3,4)), curvatures=(0.,.5))
        shapes = dict(dom=path,
            line=path.boundary['segment0'],
            arc=path.boundary['segment1'])
        topo, geom = mesh.csgmsh(shapes, elemsize=.1, order=2)
        with self.subTest('interior'):
            area = numpy.pi - 2
            self.assertAreaCentroid(topo, geom, mass=area, centroid=(3-4/3/area, 2+4/3/area), degree=2, places=6)
        with self.subTest('boundary'):
            self.assertAreaCentroid(topo.boundary['line'], geom, mass=2**1.5, centroid=(2,3), places=7)
            area = numpy.pi
            self.assertAreaCentroid(topo.boundary['arc'], geom, mass=area, centroid=(3-4/area,2+4/area), places=7)

    def test_box(self):
        'Cube with 1<x<2 and 3<y<4 and 5<z<6'
        for periodic in ('', 'x', 'y', 'z', 'yz', 'xz', 'yz', 'xyz'):
            box = cs.Box(cs.Interval(1, 2, periodic='x' in periodic), cs.Interval(3, 4, periodic='y' in periodic), cs.Interval(5, 6, periodic='z' in periodic))
            shapes = dict(dom=box,
                left=box.boundary['left'],
                right=box.boundary['right'],
                bottom=box.boundary['bottom'],
                top=box.boundary['top'],
                front=box.boundary['front'],
                back=box.boundary['back'])
            topo, geom = mesh.csgmsh(shapes, elemsize=.2)
            with self.subTest(f'interior(periodic={periodic})'):
                self.assertAreaCentroid(topo, geom, mass=1, centroid=(1.5,3.5,5.5), places=10)
            volume = 0
            if 'x' not in periodic:
                with self.subTest(f'left,right(periodic={periodic})'):
                    self.assertAreaCentroid(topo.boundary['left'], geom, mass=1, centroid=(1,3.5,5.5), places=10)
                    self.assertAreaCentroid(topo.boundary['right'], geom, mass=1, centroid=(2,3.5,5.5), places=10)
                volume += 2
            if 'y' not in periodic:
                with self.subTest(f'bottom,top(periodic={periodic})'):
                    self.assertAreaCentroid(topo.boundary['bottom'], geom, mass=1, centroid=(1.5,3,5.5), places=10)
                    self.assertAreaCentroid(topo.boundary['top'], geom, mass=1, centroid=(1.5,4,5.5), places=10)
                volume += 2
            if 'z' not in periodic:
                with self.subTest(f'front,back(periodic={periodic})'):
                    self.assertAreaCentroid(topo.boundary['front'], geom, mass=1, centroid=(1.5,3.5,5), places=10)
                    self.assertAreaCentroid(topo.boundary['back'], geom, mass=1, centroid=(1.5,3.5,6), places=10)
                volume += 2
            with self.subTest(f'boundary(periodic={periodic})'):
                if volume:
                    self.assertAreaCentroid(topo.boundary, geom, mass=volume, centroid=(1.5,3.5,5.5), places=10)
                else:
                    self.assertEqual(len(topo.boundary), 0)

    def test_sphere(self):
        'Sphere with radius .5 centered at (1,2,3)'
        sphere = cs.Sphere(center=(1,2,3), radius=.5)
        shapes = dict(dom=sphere, wall=sphere.boundary['wall'])
        topo, geom = mesh.csgmsh(shapes, elemsize=.1, order=2)
        with self.subTest(f'interior'):
            self.assertAreaCentroid(topo, geom, mass=numpy.pi/6, centroid=(1,2,3), degree=2, places=4)
        with self.subTest(f'boundary'):
            self.assertAreaCentroid(topo.boundary, geom, mass=numpy.pi, centroid=(1,2,3), degree=2, places=4)

    def test_cylinder(self):
        'Cylinder with radius .5, with front plane in (1,2,3) and back plane in (4,5,6)'
        for periodic in False, True:
            cyl = cs.Cylinder(front=(1,2,3), back=(4,5,6), radius=.5, periodic=periodic)
            shapes = dict(dom=cyl,
                side=cyl.boundary['side'],
                front=cyl.boundary['front'],
                back=cyl.boundary['back'])
            topo, geom = mesh.csgmsh(shapes, elemsize=.1, order=2)
            L = 3 * numpy.sqrt(3)
            A = numpy.pi * .5**2
            with self.subTest(f'interior(periodic={periodic})'):
                self.assertAreaCentroid(topo, geom, mass=L*A, centroid=(2.5,3.5,4.5), degree=2, places=4)
            area = numpy.pi*L
            with self.subTest(f'side(periodic={periodic})'):
                self.assertAreaCentroid(topo.boundary['side'], geom, mass=area, centroid=(2.5,3.5,4.5), degree=2, places=4)
            if not periodic:
                with self.subTest(f'front,back(periodic={periodic})'):
                    self.assertAreaCentroid(topo.boundary['front'], geom, mass=A, centroid=(1,2,3), degree=2, places=4)
                    self.assertAreaCentroid(topo.boundary['back'], geom, mass=A, centroid=(4,5,6), degree=2, places=4)
                area += 2*A
            with self.subTest('boundary'):
                self.assertAreaCentroid(topo.boundary, geom, mass=area, centroid=(2.5,3.5,4.5), degree=2, places=4)

    def test_cut_square(self):
        'Unit square with circular cutout'
        rect = cs.Rectangle()
        circ = cs.Circle(center=(1.,1.), radius=.5)
        shapes = dict(dom=rect-circ,
            circ=circ.boundary,
            left=rect.boundary['left'],
            right=rect.boundary['right'],
            bottom=rect.boundary['bottom'],
            top=rect.boundary['top'])
        topo, geom = mesh.csgmsh(shapes, elemsize=.1, order=2)
        with self.subTest('interior'):
            volume = 1 - numpy.pi/16
            c = 1 - 11/24/volume
            self.assertAreaCentroid(topo, geom, mass=volume, centroid=(c,c), degree=2, places=5)
        with self.subTest('boundary'):
            self.assertAreaCentroid(topo.boundary['left'], geom, mass=1, centroid=(0,.5), places=10)
            self.assertAreaCentroid(topo.boundary['right'], geom, mass=.5, centroid=(1,.25), places=10)
            self.assertAreaCentroid(topo.boundary['bottom'], geom, mass=1, centroid=(.5,0), places=10)
            self.assertAreaCentroid(topo.boundary['top'], geom, mass=.5, centroid=(.25,1), places=10)
            self.assertAreaCentroid(topo.boundary['circ'], geom, mass=numpy.pi*.5*2/4, centroid=(1-1/numpy.pi,1-1/numpy.pi), places=6)

    def test_fused(self):
        'Unit square fused with a circle'
        rect = cs.Rectangle()
        circ = cs.Circle(center=(1.,.5), radius=.5)
        topo, geom = mesh.csgmsh(dict(dom=rect|circ), elemsize=.1, order=2)
        #topo, geom = mesh.csgmsh(dict(dom=rect|circ), elemsize=.1, order=2)
        with self.subTest('interior'):
            volume = 1 + numpy.pi/8
            self.assertAreaCentroid(topo, geom, mass=volume, centroid=((7/12+numpy.pi/8)/volume,.5), degree=2, places=5)
        with self.subTest('boundary'):
            area = 3 + numpy.pi/2
            self.assertAreaCentroid(topo.boundary, geom, mass=area, centroid=((3/2+numpy.pi/2)/area,.5), places=5)

    def test_revolved2(self):
        orig = numpy.array([2., 3.])
        line = cs.Interval(1,2)
        rev = line.revolved(origin=orig, xaxis=(1,-1), angle=90)
        # xaxis is rotated from (1,-1) to (1,1)
        shapes = dict(dom=rev,
            front=rev.boundary['front'],
            back=rev.boundary['back'],
            left=rev.boundary['side-left'],
            right=rev.boundary['side-right'])
        topo, geom = mesh.csgmsh(shapes, elemsize=.1, order=2)
        with self.subTest('interior'):
            volume = numpy.pi * 3 / 4
            self.assertAreaCentroid(topo, geom, mass=volume, centroid=orig+((7/3)*numpy.sqrt(2)/volume,0), degree=2, places=6)
        with self.subTest('boundary'):
            area = numpy.pi * 3 / 2 + 2
            self.assertAreaCentroid(topo.boundary, geom, mass=area, centroid=orig+((13/2)*numpy.sqrt(2)/area,0), degree=2, places=6)
            self.assertAreaCentroid(topo.boundary['front'], geom, mass=1, centroid=orig+(numpy.sqrt(2)*3/4,-numpy.sqrt(2)*3/4), degree=2, places=6)
            self.assertAreaCentroid(topo.boundary['back'], geom, mass=1, centroid=orig+(numpy.sqrt(2)*3/4,numpy.sqrt(2)*3/4), degree=2, places=6)
            self.assertAreaCentroid(topo.boundary['left'], geom, mass=numpy.pi/2, centroid=orig+(2/numpy.pi*numpy.sqrt(2),0), degree=2, places=6)
            self.assertAreaCentroid(topo.boundary['right'], geom, mass=numpy.pi, centroid=orig+(4/numpy.pi*numpy.sqrt(2),0), degree=2, places=6)

    def test_revolved3(self):
        'Quarter annular sector with inner radius 1 and outer radius 2, symmetric in z=0'
        orig = numpy.array([2., 3., 4.])
        rect = cs.Rectangle(x=cs.Interval(1,2))
        rev = rect.revolved(origin=orig, xaxis=(-1,0,1), yaxis=(0,1,0), angle=-90)
        # x-axis is rotated from (-1,0,1) to (-1,0,-1)
        shapes = dict(dom=rev,
            front=rev.boundary['front'],
            back=rev.boundary['back'],
            left=rev.boundary['side-left'],
            right=rev.boundary['side-right'],
            bottom=rev.boundary['side-bottom'],
            top=rev.boundary['side-top'])
        topo, geom = mesh.csgmsh(shapes, elemsize=.1, order=2)
        with self.subTest('interior'):
            volume = numpy.pi * 3 / 4
            cx = -(7/3)*numpy.sqrt(2)/volume
            self.assertAreaCentroid(topo, geom, mass=volume, centroid=orig+(cx,.5,0), degree=2, places=6)
        with self.subTest('boundary'):
            area = numpy.pi * 3 + 2
            self.assertAreaCentroid(topo.boundary, geom, mass=area, centroid=orig+(-(67/6)*numpy.sqrt(2)/area,.5,0), degree=2, places=6)
            self.assertAreaCentroid(topo.boundary['front'], geom, mass=1, centroid=orig+(-(3/4)*numpy.sqrt(2),.5,(3/4)*numpy.sqrt(2)), degree=2, places=6)
            self.assertAreaCentroid(topo.boundary['back'], geom, mass=1, centroid=orig+(-(3/4)*numpy.sqrt(2),.5,-(3/4)*numpy.sqrt(2)), degree=2, places=6)
            self.assertAreaCentroid(topo.boundary['left'], geom, mass=.5*numpy.pi, centroid=orig+(-2/numpy.pi*numpy.sqrt(2),.5,0), degree=2, places=6)
            self.assertAreaCentroid(topo.boundary['right'], geom, mass=numpy.pi, centroid=orig+(-4/numpy.pi*numpy.sqrt(2),.5,0), degree=2, places=6)
            self.assertAreaCentroid(topo.boundary['top'], geom, mass=volume, centroid=orig+(cx,1,0), degree=2, places=6)
            self.assertAreaCentroid(topo.boundary['bottom'], geom, mass=volume, centroid=orig+(cx,0,0), degree=2, places=6)

    def test_revolved_full(self):
        'Annular ring with inner radius 1 and outer radius 2'
        orig = numpy.array([2., 3., 4.])
        rect = cs.Rectangle(x=cs.Interval(1,2))
        rev = rect.revolved(origin=orig, xaxis=(-1,0,1), yaxis=(0,1,0), angle=360)
        shapes = dict(dom=rev,
            left=rev.boundary['side-left'],
            right=rev.boundary['side-right'],
            bottom=rev.boundary['side-bottom'],
            top=rev.boundary['side-top'])
        topo, geom = mesh.csgmsh(shapes, elemsize=.1, order=2)
        with self.subTest('interior'):
            self.assertAreaCentroid(topo, geom, mass=3*numpy.pi, centroid=orig+(0,.5,0), degree=2, places=6)
        with self.subTest('boundary'):
            self.assertAreaCentroid(topo.boundary, geom, mass=12*numpy.pi, centroid=orig+(0,.5,0), degree=2, places=5)
            self.assertAreaCentroid(topo.boundary['left'], geom, mass=2*numpy.pi, centroid=orig+(0,.5,0), degree=2, places=5)
            self.assertAreaCentroid(topo.boundary['right'], geom, mass=4*numpy.pi, centroid=orig+(0,.5,0), degree=2, places=5)
            self.assertAreaCentroid(topo.boundary['bottom'], geom, mass=3*numpy.pi, centroid=orig+(0,0,0), degree=2, places=5)
            self.assertAreaCentroid(topo.boundary['top'], geom, mass=3*numpy.pi, centroid=orig+(0,1,0), degree=2, places=5)

    def test_pipe2(self):
        orig = numpy.array([2., 3.])
        line = cs.Interval(0, 2)
        pipe = line.extruded(segments=[[numpy.pi/2,1], [1,0], [2*numpy.pi,-.25]], origin=orig)
        shapes = dict(dom=pipe,
            front=pipe.boundary['front'],
            back=pipe.boundary['back'],
            left0=pipe.boundary['segment0-left'],
            right0=pipe.boundary['segment0-right'],
            left1=pipe.boundary['segment1-left'],
            right1=pipe.boundary['segment1-right'],
            left2=pipe.boundary['segment2-left'],
            right2=pipe.boundary['segment2-right'])
        topo, geom = mesh.csgmsh(shapes, elemsize=.1, order=2)
        with self.subTest('interior'):
            V = numpy.pi * 5 + 2
            M = -8 * numpy.pi - 13, 15 * numpy.pi - 6
            self.assertAreaCentroid(topo, geom, mass=V, centroid=orig + numpy.divide(M,V), degree=2, places=6)
        with self.subTest('boundary'):
            self.assertAreaCentroid(topo.boundary['front'], geom, mass=2, centroid=orig+(1,0), degree=1, places=9)
            self.assertAreaCentroid(topo.boundary['left0'], geom, mass=numpy.pi/2, centroid=orig+(-1+2/numpy.pi, 2/numpy.pi), degree=2, places=6)
            self.assertAreaCentroid(topo.boundary['right0'], geom, mass=numpy.pi*3/2, centroid=orig+(-1+6/numpy.pi, 6/numpy.pi), degree=2, places=6)
            self.assertAreaCentroid(topo.boundary['left1'], geom, mass=1, centroid=orig+(-1.5,1), degree=1, places=9)
            self.assertAreaCentroid(topo.boundary['right1'], geom, mass=1, centroid=orig+(-1.5,3), degree=1, places=9)
            self.assertAreaCentroid(topo.boundary['left2'], geom, mass=numpy.pi*2, centroid=orig+(-2-8/numpy.pi, 5-8/numpy.pi), degree=2, places=6)
            self.assertAreaCentroid(topo.boundary['right2'], geom, mass=numpy.pi, centroid=orig+(-2-4/numpy.pi,5-4/numpy.pi), degree=2, places=6)
            self.assertAreaCentroid(topo.boundary['back'], geom, mass=2, centroid=orig+(-5,5), degree=1, places=9)

    def test_pipe3(self):
        orig = numpy.array([2., 3., 4.])
        rect = cs.Rectangle(y=cs.Interval(0, 2))
        pipe = rect.extruded(segments=[[numpy.pi/2,1,0], [1,0,0], [numpy.pi,0,.5]], origin=orig)
        shapes = dict(dom=pipe,
            front=pipe.boundary['front'],
            back=pipe.boundary['back'],
            left0=pipe.boundary['segment0-left'],
            right0=pipe.boundary['segment0-right'],
            bottom0=pipe.boundary['segment0-bottom'],
            top0=pipe.boundary['segment0-top'],
            left1=pipe.boundary['segment1-left'],
            right1=pipe.boundary['segment1-right'],
            bottom1=pipe.boundary['segment1-bottom'],
            top1=pipe.boundary['segment1-top'],
            left2=pipe.boundary['segment2-left'],
            right2=pipe.boundary['segment2-right'],
            bottom2=pipe.boundary['segment2-bottom'],
            top2=pipe.boundary['segment2-top'])
        topo, geom = mesh.csgmsh(shapes, elemsize=.1, order=2)
        with self.subTest('interior'):
            V = numpy.pi * (7/2) + 2
            M = 4 * numpy.pi - 11/3, 1 - 5 * numpy.pi, 38/3 + numpy.pi * 3
            self.assertAreaCentroid(topo, geom, mass=V, centroid=orig + numpy.divide(M,V), degree=2, places=6)
        with self.subTest('boundary'):
            self.assertAreaCentroid(topo.boundary['front'], geom, mass=2, centroid=orig+(.5,1,0), degree=1, places=9)
            self.assertAreaCentroid(topo.boundary['left0'], geom, mass=numpy.pi*2, centroid=orig+(0,13/3/numpy.pi-1,13/3/numpy.pi), degree=2, places=6)
            self.assertAreaCentroid(topo.boundary['right0'], geom, mass=numpy.pi*2, centroid=orig+(1,13/3/numpy.pi-1,13/3/numpy.pi), degree=2, places=6)
            self.assertAreaCentroid(topo.boundary['top0'], geom, mass=numpy.pi*3/2, centroid=orig+(.5,6/numpy.pi-1,6/numpy.pi), degree=2, places=6)
            self.assertAreaCentroid(topo.boundary['bottom0'], geom, mass=numpy.pi/2, centroid=orig+(.5,2/numpy.pi-1,2/numpy.pi), degree=2, places=6)
            self.assertAreaCentroid(topo.boundary['left1'], geom, mass=2, centroid=orig+(0,-1.5,2), degree=1, places=9)
            self.assertAreaCentroid(topo.boundary['right1'], geom, mass=2, centroid=orig+(1,-1.5,2), degree=1, places=9)
            self.assertAreaCentroid(topo.boundary['top1'], geom, mass=1, centroid=orig+(.5,-1.5,3), degree=1, places=9)
            self.assertAreaCentroid(topo.boundary['bottom1'], geom, mass=1, centroid=orig+(.5,-1.5,1), degree=1, places=9)
            self.assertAreaCentroid(topo.boundary['left2'], geom, mass=numpy.pi*2, centroid=orig+(2-4/numpy.pi,-2-4/numpy.pi,2), degree=2, places=6)
            self.assertAreaCentroid(topo.boundary['right2'], geom, mass=numpy.pi, centroid=orig+(2-2/numpy.pi,-2-2/numpy.pi,2), degree=2, places=6)
            self.assertAreaCentroid(topo.boundary['top2'], geom, mass=numpy.pi*3/4, centroid=orig+(2-28/9/numpy.pi,-2-28/9/numpy.pi,3), degree=2, places=6)
            self.assertAreaCentroid(topo.boundary['bottom2'], geom, mass=numpy.pi*3/4, centroid=orig+(2-28/9/numpy.pi,-2-28/9/numpy.pi,1), degree=2, places=6)
            self.assertAreaCentroid(topo.boundary['back'], geom, mass=2, centroid=orig+(2,-3.5,2), degree=1, places=9)

    def test_inclusion(self):
        outer = cs.Box(cs.Interval(-2, 2), cs.Interval(-2, 2), cs.Interval(-2, 2))
        inner = cs.Box(cs.Interval(-1, 1), cs.Interval(-1, 1), cs.Interval(-1, 1))
        shapes = dict(dom=outer-inner,
            innerleft=inner.boundary['left'],
            innerright=inner.boundary['right'],
            innertop=inner.boundary['top'],
            innerbottom=inner.boundary['bottom'],
            innerfront=inner.boundary['front'],
            innerback=inner.boundary['back'],
            outerleft=outer.boundary['left'],
            outerright=outer.boundary['right'],
            outertop=outer.boundary['top'],
            outerbottom=outer.boundary['bottom'],
            outerfront=outer.boundary['front'],
            outerback=outer.boundary['back'])
        topo, geom = mesh.csgmsh(shapes, elemsize=.5, order=2)
        with self.subTest('interior'):
            self.assertAreaCentroid(topo, geom, mass=56, centroid=(0,0,0), degree=1, places=13)
        with self.subTest('boundary'):
            self.assertAreaCentroid(topo.boundary['innerleft'], geom, mass=4, centroid=(-1,0,0), degree=1, places=13)
            self.assertAreaCentroid(topo.boundary['innerright'], geom, mass=4, centroid=(1,0,0), degree=1, places=13)
            self.assertAreaCentroid(topo.boundary['innerbottom'], geom, mass=4, centroid=(0,-1,0), degree=1, places=13)
            self.assertAreaCentroid(topo.boundary['innertop'], geom, mass=4, centroid=(0,1,0), degree=1, places=13)
            self.assertAreaCentroid(topo.boundary['innerfront'], geom, mass=4, centroid=(0,0,-1), degree=1, places=13)
            self.assertAreaCentroid(topo.boundary['innerback'], geom, mass=4, centroid=(0,0,1), degree=1, places=13)
            self.assertAreaCentroid(topo.boundary['outerleft'], geom, mass=16, centroid=(-2,0,0), degree=1, places=13)
            self.assertAreaCentroid(topo.boundary['outerright'], geom, mass=16, centroid=(2,0,0), degree=1, places=13)
            self.assertAreaCentroid(topo.boundary['outerbottom'], geom, mass=16, centroid=(0,-2,0), degree=1, places=13)
            self.assertAreaCentroid(topo.boundary['outertop'], geom, mass=16, centroid=(0,2,0), degree=1, places=13)
            self.assertAreaCentroid(topo.boundary['outerfront'], geom, mass=16, centroid=(0,0,-2), degree=1, places=13)
            self.assertAreaCentroid(topo.boundary['outerback'], geom, mass=16, centroid=(0,0,2), degree=1, places=13)


class TestAxes2(unittest.TestCase):

    def test_eye(self):
        numpy.testing.assert_almost_equal(cs._axes.Axes2.eye()[:], numpy.eye(2))

    def test_from_x(self):
        xaxis = numpy.array([-1.,1.])
        axes = cs._axes.Axes2.from_x(xaxis)
        numpy.testing.assert_almost_equal(axes[0], xaxis / numpy.linalg.norm(xaxis))

    def test_rotate(self):
        axes = cs._axes.Axes2.from_x([1,0]).rotate(numpy.pi/2)
        numpy.testing.assert_almost_equal(axes[0], [0,1])


class TestAxes3(unittest.TestCase):

    def test_eye(self):
        numpy.testing.assert_almost_equal(cs._axes.Axes3.eye()[:], numpy.eye(3))

    def test_from_xy(self):
        xaxis = numpy.array([1.,2.,4.])
        yaxis = numpy.array([2.,1.,-1.]) # orthogonal
        axes = cs._axes.Axes3.from_xy(xaxis, yaxis)
        numpy.testing.assert_almost_equal(axes[0], xaxis / numpy.linalg.norm(xaxis))
        numpy.testing.assert_almost_equal(axes[1], yaxis / numpy.linalg.norm(yaxis))

    def test_from_rotation_vector(self):
        v = numpy.array([.3,.4,.5])
        axes = cs._axes.Axes3.from_rotation_vector(v)
        numpy.testing.assert_almost_equal(v, axes.rotation_vector)

    def test_rotate(self):
        eye = cs._axes.Axes3.eye()
        numpy.testing.assert_almost_equal(eye.rotate([0,0,numpy.pi/2])[:], [[0,1,0],[-1,0,0],[0,0,1]])
        numpy.testing.assert_almost_equal(eye.rotate([0,numpy.pi/2,0])[:], [[0,0,-1],[0,1,0],[1,0,0]])
        numpy.testing.assert_almost_equal(eye.rotate([numpy.pi/2,0,0])[:], [[1,0,0],[0,0,1],[0,-1,0]])


class TestInterval(unittest.TestCase):

    def test_left_right(self):
        iv = cs.Interval(left=1, right=3)
        self.assertAlmostEqual(iv.left, 1)
        self.assertAlmostEqual(iv.right, 3)
        self.assertAlmostEqual(iv.center, 2)
        self.assertAlmostEqual(iv.length, 2)

    def test_center_length(self):
        iv = cs.Interval(center=1, length=4)
        self.assertAlmostEqual(iv.left, -1)
        self.assertAlmostEqual(iv.right, 3)
        self.assertAlmostEqual(iv.center, 1)
        self.assertAlmostEqual(iv.length, 4)
