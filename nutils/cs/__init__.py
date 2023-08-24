'''Constructive Solid Geometry helper for Gmsh'''


from ._shape import (
    Box,
    Circle,
    Cylinder,
    Ellipse,
    Entities,
    Interval,
    Line,
    Path,
    Point,
    Rectangle,
    Skeleton,
    Sphere,
    generate_mesh,
)


from ._field import (
    AsField,
    Ball,
    LocalRefinement,
    Max,
    Min,
    set_background,
    x,
    y,
    z,
)


def write(fname: str, entities: Entities, elemsize: AsField, order: int = 1) -> None:
    'Create .msh file based on Constructive Solid Geometry description.'

    import treelog, gmsh # type: ignore

    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        gmsh.option.setNumber('Mesh.Binary', 1)
        gmsh.option.setNumber('Mesh.ElementOrder', order)
        gmsh.option.setNumber('Mesh.CharacteristicLengthExtendFromBoundary', 0)
        gmsh.option.setNumber('Mesh.CharacteristicLengthFromPoints', 0)
        gmsh.option.setNumber('Mesh.CharacteristicLengthFromCurvature', 0)

        gmsh.logger.start()
        try:
            set_background(gmsh.model.mesh.field, elemsize)
            generate_mesh(gmsh.model, entities)
            gmsh.write(fname)
        finally:
            with treelog.context('gmsh'):
                for line in gmsh.logger.get():
                    level, sep, msg = line.partition(': ')
                    if level in ('Debug', 'Info', 'Warning', 'Error'):
                        getattr(treelog, level.lower())(msg)
            gmsh.logger.stop()

    finally:
        gmsh.finalize()


# vim:sw=4:sts=4:et
