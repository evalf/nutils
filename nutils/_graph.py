from typing import Mapping, MutableMapping, Optional, Iterator, Iterable, Generator, Sequence, List, MutableSet, Callable, Tuple, Generic, TypeVar, Dict
import typing
import itertools
import treelog
import subprocess
import abc
import html
import re
import shutil

Metadata = TypeVar('Metadata')
GraphvizColorCallback = Callable[['Node'], Optional[str]]


class Subgraph:

    def __init__(self, label: str, parent: Optional['Subgraph'] = None) -> None:
        self.label = label
        self.parent = parent


class Node(Generic[Metadata], metaclass=abc.ABCMeta):

    def __init__(self, metadata: Metadata, subgraph: Optional[Subgraph] = None) -> None:
        self.metadata = metadata
        self.subgraph = subgraph

    @abc.abstractmethod
    def __bool__(self) -> bool:
        raise NotImplementedError  # pragma: no cover

    @abc.abstractmethod
    def _generate_asciitree_nodes(self, cache: MutableMapping['Node[Metadata]', str], id_gen_map: Mapping[Optional[Subgraph], Iterator[str]], select: str, bridge: str) -> Generator[str, None, None]:
        raise NotImplementedError  # pragma: no cover

    @abc.abstractmethod
    def _collect_graphviz_nodes_edges(self, cache: MutableMapping['Node[Metadata]', str], id_gen: Iterator[str], nodes: MutableMapping[Optional[Subgraph], List[str]], edges: List[str], parent_subgraph: Optional[Subgraph], fill_color: Optional[GraphvizColorCallback] = None) -> Optional[str]:
        raise NotImplementedError  # pragma: no cover

    def walk(self, seen: MutableSet['Node[Metadata]']) -> Iterator['Node[Metadata]']:
        raise NotImplementedError  # pragma: no cover

    def generate_asciitree(self, richoutput: bool = False) -> str:
        subgraph_children = _collect_subgraphs(self)
        if len(subgraph_children) > 1:
            id_gen_map = {}  # type: Dict[Optional[Subgraph], Iterator[str]]
            parts = ['SUBGRAPHS\n'], _generate_asciitree_subgraphs(subgraph_children, id_gen_map, None, '', ''), ['NODES\n']  # type: Sequence[Iterable[str]]
        else:
            id_gen_map = {None: (f'%{i}' for i in itertools.count())}
            parts = []
        asciitree = ''.join(itertools.chain(*parts, self._generate_asciitree_nodes({}, id_gen_map, '', '')))
        if not richoutput:
            asciitree = asciitree.replace('├', ':').replace('└', ':').replace('│', '|')
        return asciitree

    def generate_graphviz_source(self, *, fill_color: Optional[GraphvizColorCallback] = None) -> str:
        edges = []  # type: List[str]
        nodes = {}  # type: Dict[Optional[Subgraph], List[str]]
        subgraph_children = _collect_subgraphs(self)
        id_gen = map(str, itertools.count())
        self._collect_graphviz_nodes_edges({}, id_gen, nodes, edges, None, fill_color)
        return ''.join(itertools.chain(['digraph {bgcolor="darkgray";'], _generate_graphviz_subgraphs(subgraph_children, nodes, None, id_gen, 0), edges, ['}']))

    def export_graphviz(self, *, fill_color: Optional[GraphvizColorCallback] = None, dot_path: str = 'dot', image_type: str = 'svg') -> None:
        dot = shutil.which(dot_path)
        if not dot:
            raise RuntimeError(f'cannot find graphviz application {dot_path!r}')
        src = self.generate_graphviz_source(fill_color=fill_color)
        with treelog.infofile('dot.'+image_type, 'wb') as img:
            src = src.replace(';', ';\n')
            status = subprocess.run([dot, '-Gstart=1', '-T'+image_type], input=src.encode(), stdout=subprocess.PIPE)
            if status.returncode:
                for i, line in enumerate(src.split('\n'), 1):
                    print('{:04d}  {}'.format(i, line))
                treelog.warning('graphviz failed for error code', status.returncode)
            graph = status.stdout
            if image_type == 'svg':
                graph = re.sub(rb'<svg width="(\d+)pt" height="(\d+)pt"', lambda match:
                    b'<svg width="%dpt" height="%dpt"' % tuple((int(l)*2)//3 for l in match.groups()), graph, count=1)
                i = graph.rindex(b'</svg>')
                graph = graph[:i] + clickHandler + graph[i:]
            img.write(graph)

clickHandler = b'''
<style>
g.edge, g.node { cursor: pointer; }
.highlight path { stroke: orange; stroke-width: 2; }
.highlight polygon:first-of-type { fill: orange; }
.highlight polygon { stroke: orange; }
.highlight text { fill: white; }
</style>
<script>
document.addEventListener("click", function (event) {
    const g = event.target.closest("g");
    if (g.classList.contains("edge")) {
        g.classList.toggle("highlight"); }
    else if (g.classList.contains("node")) {
        const index = g.firstElementChild.textContent;
        const pattern = new RegExp("(^|[^0-9])" + index + "($|[^0-9])");
        const isnew = g.classList.toggle("highlight");
        for (const edge of document.getElementsByClassName("edge")) {
            if (pattern.test(edge.firstElementChild.textContent)) {
                edge.classList.toggle("highlight", isnew); } } } });
</script>'''


class RegularNode(Node[Metadata]):

    def __init__(self, label: str, args: Sequence[Node[Metadata]], kwargs: Mapping[str, Node[Metadata]], metadata: Metadata, subgraph: Optional[Subgraph] = None) -> None:
        self._label = label
        self._args = tuple(args)
        self._kwargs = dict(kwargs)
        super().__init__(metadata, subgraph)

    def __bool__(self) -> bool:
        return True

    def _generate_asciitree_nodes(self, cache: MutableMapping[Node[Metadata], str], id_gen_map: Mapping[Optional[Subgraph], str], select: str, bridge: str) -> Generator[str, None, None]:
        if self in cache:
            yield '{}{}\n'.format(select, cache[self])
        else:
            cache[self] = id = next(id_gen_map[self.subgraph])
            yield '{}{} = {}\n'.format(select, id, self._label.replace('\n', '; '))
            args = tuple(('', arg) for arg in self._args if arg) + tuple(('{} = '.format(name), arg) for name, arg in self._kwargs.items())
            for i, (prefix, arg) in enumerate(args, 1-len(args)):
                yield from arg._generate_asciitree_nodes(cache, id_gen_map, bridge+('├ ' if i else '└ ')+prefix, bridge+('│ ' if i else '  '))

    def _collect_graphviz_nodes_edges(self, cache: MutableMapping[Node[Metadata], str], id_gen: Iterator[str], nodes: MutableMapping[Optional[Subgraph], List[str]], edges: List[str], parent_subgraph: Optional[Subgraph], fill_color: Optional[GraphvizColorCallback] = None) -> Optional[str]:
        if self in cache:
            return cache[self]
        cache[self] = id = next(id_gen)
        if self._kwargs:
            table = ['<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0">']
            table += ['<TR>', *('<TD PORT="kwarg{}"><FONT POINT-SIZE="10">{}</FONT></TD>'.format(ikwarg, html.escape(name)) for ikwarg, name in enumerate(self._kwargs)), '</TR>']
            table += ['<TR><TD COLSPAN="{}">{}</TD></TR>'.format(len(self._kwargs), html.escape(line)) for line in self._label.split('\n')]
            table += ['</TABLE>']
            attributes = ['shape=plain', 'label=<{}>'.format(''.join(table))]
        else:
            attributes = ['shape=box', 'label="{}"'.format(self._label.replace('"', '\\"'))]
        attributes.extend(_graphviz_fill_color_attributes(self, fill_color))
        nodes.setdefault(self.subgraph, []).append('{} [{}];'.format(id, ','.join(attributes)))
        for arg in self._args:
            arg_id = arg._collect_graphviz_nodes_edges(cache, id_gen, nodes, edges, self.subgraph, fill_color)
            if arg_id:
                edges.append('{} -> {};'.format(arg_id, id))
        for ikwarg, arg in enumerate(self._kwargs.values()):
            arg_id = arg._collect_graphviz_nodes_edges(cache, id_gen, nodes, edges, self.subgraph, fill_color)
            if arg_id:
                edges.append('{} -> {}:kwarg{}:n;'.format(arg_id, id, ikwarg))
        return id

    def walk(self, seen: MutableSet[Node[Metadata]]) -> Iterator[Node[Metadata]]:
        if self in seen:
            return
        seen.add(self)
        yield self
        for arg in self._args:
            yield from arg.walk(seen)
        for arg in self._kwargs.values():
            yield from arg.walk(seen)


class TupleNode(RegularNode[Metadata]):

    def __init__(self, items: Tuple[Node, ...], metadata: Metadata, subgraph: Optional[Subgraph] = None) -> None:
        self.items = items
        super().__init__(label='Tuple', args=items, kwargs={}, metadata=metadata, subgraph=subgraph)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Node:
        return self.items[index]


class DuplicatedLeafNode(Node[Metadata]):

    def __init__(self, label: str, metadata: Metadata) -> None:
        self._label = label
        super().__init__(metadata)

    def __bool__(self) -> bool:
        return True

    def _generate_asciitree_nodes(self, cache: MutableMapping[Node[Metadata], str], id_gen_map: Mapping[Optional[Subgraph], str], select: str, bridge: str) -> Generator[str, None, None]:
        yield '{}{}\n'.format(select, self._label.replace('\n', '; '))

    def _collect_graphviz_nodes_edges(self, cache: MutableMapping[Node[Metadata], str], id_gen: Iterator[str], nodes: MutableMapping[Optional[Subgraph], List[str]], edges: List[str], parent_subgraph: Optional[Subgraph], fill_color: Optional[GraphvizColorCallback] = None) -> Optional[str]:
        id = next(id_gen)
        attributes = ['shape=box', 'label="{}"'.format(self._label.replace('"', '\\"')), *_graphviz_fill_color_attributes(self, fill_color)]
        nodes.setdefault(parent_subgraph, []).append('{} [{}];'.format(id, ','.join(attributes)))
        return id

    def walk(self, seen: MutableSet[Node[Metadata]]) -> Iterator[Node[Metadata]]:
        if self in seen:
            return
        seen.add(self)
        yield self


class InvisibleNode(Node[Metadata]):

    def __init__(self, metadata: Metadata) -> None:
        super().__init__(metadata)

    def __bool__(self) -> bool:
        return False

    def _generate_asciitree_nodes(self, cache: MutableMapping[Node[Metadata], str], id_gen_map: Mapping[Optional[Subgraph], str], select: str, bridge: str) -> Generator[str, None, None]:
        yield '{}\n'.format(select)

    def _collect_graphviz_nodes_edges(self, cache: MutableMapping[Node[Metadata], str], id_gen: Iterator[str], nodes: MutableMapping[Optional[Subgraph], List[str]], edges: List[str], parent_subgraph: Optional[Subgraph], fill_color: Optional[GraphvizColorCallback] = None) -> Optional[str]:
        return None

    def walk(self, seen: MutableSet[Node[Metadata]]) -> Iterator[Node[Metadata]]:
        if self in seen:
            return
        seen.add(self)
        yield self


def _graphviz_fill_color_attributes(node: Node[Metadata], fill_color: Optional[GraphvizColorCallback]) -> Sequence[str]:
    if not fill_color:
        return ()
    value = fill_color(node)
    if value is None:
        return ()
    return 'style=filled', 'fillcolor="{}"'.format(value)


def _collect_subgraphs(node: Node[Metadata]) -> Dict[Optional[Subgraph], List[Subgraph]]:
    children = {None: []}  # type: Dict[Optional[Subgraph], List[Subgraph]]
    for node in node.walk(set()):
        subgraph = node.subgraph
        if subgraph and subgraph not in children:
            children[subgraph] = []
            while subgraph and subgraph.parent not in children:
                children[subgraph.parent] = [subgraph]
                subgraph = subgraph.parent
            subgraph = typing.cast(Subgraph, subgraph)
            children[subgraph.parent].append(subgraph)
    return children


def _generate_asciitree_subgraphs(children: Mapping[Optional[Subgraph], Sequence[Subgraph]], id_gen_map: MutableMapping[Optional[Subgraph], Iterator[str]], subgraph: Optional[Subgraph], select: str, bridge: str) -> Iterator[str]:
    assert subgraph not in id_gen_map
    id = chr(ord('A') + len(id_gen_map))
    id_gen_map[subgraph] = (f'%{id}{i}' for i in itertools.count())
    if subgraph:
        yield '{}{} = {}\n'.format(select, id, subgraph.label.replace('\n', '; '))
    else:
        yield '{}{}\n'.format(select, id)
    for i, child in enumerate(children[subgraph], 1-len(children[subgraph])):
        yield from _generate_asciitree_subgraphs(children, id_gen_map, child, bridge+('├ ' if i else '└ '), bridge+('│ ' if i else '  '))


def _generate_graphviz_subgraphs(children: Mapping[Optional[Subgraph], Sequence[Subgraph]], nodes: Mapping[Optional[Subgraph], Sequence[str]], subgraph: Optional[Subgraph], id_gen: Iterator[str], depth: int) -> Iterator[str]:
    for child in children[subgraph]:
        yield 'subgraph cluster{} {{bgcolor="{}";color="none";'.format(next(id_gen), 'darkgray' if depth % 2 else 'lightgray')
        yield from _generate_graphviz_subgraphs(children, nodes, child, id_gen, depth + 1)
        yield '}'
    yield from nodes.get(subgraph, ())
