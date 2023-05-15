from nutils.testing import TestCase
from nutils import _graph
import itertools
import unittest
import sys


class DummyNode(_graph.Node):

    def __init__(self, label='', metadata=None):
        assert '\n' not in label
        self.label = label
        super().__init__(metadata)

    def __bool__(self):
        return bool(self.label)

    def _generate_asciitree_nodes(self, cache, graph_ids, select, bridge):
        yield '{}{}\n'.format(select, self.label)

    def _collect_graphviz_nodes_edges(self, cache, id_gen, nodes, edges, parent_graph, fill_color=None):
        if self:
            id = next(id_gen)
            nodes.setdefault(parent_graph, []).append('{} [label="{}"];'.format(id, self.label))
            return id

    def walk(self, seen):
        yield self


class RegularNode(TestCase):

    def test_truthiness(self):
        self.assertTrue(_graph.RegularNode('test', (), {}, 'meta'))

    @unittest.skipIf(sys.version_info < (3, 6), 'test requires dict with insertion order')
    def test_generate_asciitree_nodes(self):
        args = DummyNode('a'), DummyNode('b'), DummyNode()
        kwargs = dict(spam=DummyNode('d'), eggs=DummyNode('e'))
        node = _graph.RegularNode('test', args, kwargs, 'meta')
        cache = {}
        graph_ids = {None: (f'%X{i}' for i in itertools.count())}
        with self.subTest('first'):
            self.assertEqual(list(node._generate_asciitree_nodes(cache, graph_ids, 'S', 'B')), [
                             'S%X0 = test\n',
                             'B├ a\n',
                             'B├ b\n',
                             'B├ spam = d\n',
                             'B└ eggs = e\n'])
        with self.subTest('second'):
            self.assertEqual(list(node._generate_asciitree_nodes(cache, graph_ids, 'S', 'B')), [
                             'S%X0\n'])

    def test_collect_graphviz_nodes_edges_args(self):
        args = DummyNode('a'), DummyNode('b'), DummyNode()
        node = _graph.RegularNode('test', args, {}, 'meta')
        cache = {}
        nodes = {}
        edges = []
        cnt = map(str, itertools.count())
        for sub in 'first', 'second':
            with self.subTest(sub):
                self.assertEqual(node._collect_graphviz_nodes_edges(cache, cnt, nodes, edges, _graph.Subgraph('sub'), None), '0')
                self.assertEqual(edges, ['1 -> 0;', '2 -> 0;'])
                self.assertEqual(nodes, {None: ['0 [shape=box,label="test"];', '1 [label="a"];', '2 [label="b"];']})

    @unittest.skipIf(sys.version_info < (3, 6), 'requires dict with insertion order')
    def test_collect_graphviz_nodes_edges_mixed(self):
        args = DummyNode('a'), DummyNode('b'), DummyNode()
        kwargs = dict(spam=DummyNode('d'), eggs=DummyNode('e'))
        node = _graph.RegularNode('test', args, kwargs, 'meta')
        label = (
            '<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0">'
            '<TR>'
            '<TD PORT="kwarg0"><FONT POINT-SIZE="10">spam</FONT></TD>'
            '<TD PORT="kwarg1"><FONT POINT-SIZE="10">eggs</FONT></TD>'
            '</TR>'
            '<TR><TD COLSPAN="2">test</TD></TR>'
            '</TABLE>')
        cache = {}
        nodes = {}
        edges = []
        cnt = map(str, itertools.count())
        for sub in 'first', 'second':
            with self.subTest(sub):
                self.assertEqual(node._collect_graphviz_nodes_edges(cache, cnt, nodes, edges, _graph.Subgraph('sub'), None), '0')
                self.assertEqual(edges, ['1 -> 0;', '2 -> 0;', '3 -> 0:kwarg0:n;', '4 -> 0:kwarg1:n;'])
                self.assertEqual(nodes, {None: ['0 [shape=plain,label=<{}>];'.format(label), '1 [label="a"];', '2 [label="b"];', '3 [label="d"];', '4 [label="e"];']})

    def test_graphviz_fill_color(self):
        args = DummyNode('a'), DummyNode('b'), DummyNode()
        node = _graph.RegularNode('test', args, {}, 'meta')
        with self.subTest('some'):
            nodes = {}
            node._collect_graphviz_nodes_edges({}, map(str, itertools.count()), nodes, [], None, lambda n: 'red')
            self.assertEqual(nodes[None][0], '0 [shape=box,label="test",style=filled,fillcolor="red"];')
        with self.subTest('none'):
            nodes = {}
            node._collect_graphviz_nodes_edges({}, map(str, itertools.count()), nodes, [], None, lambda n: None)
            self.assertEqual(nodes[None][0], '0 [shape=box,label="test"];')

    @unittest.skipIf(sys.version_info < (3, 6), 'test requires dict with insertion order')
    def test_walk(self):
        args = DummyNode('a'), DummyNode('b'), DummyNode()
        kwargs = dict(spam=DummyNode('d'), eggs=DummyNode('e'))
        node = _graph.RegularNode('test', args, kwargs, 'meta')
        seen = set()
        self.assertEqual(list(node.walk(seen)), [node, *args, *kwargs.values()])
        self.assertEqual(list(node.walk(seen)), [])


class DuplicatedLeafNode(TestCase):

    def setUp(self):
        super().setUp()
        self.subgraph = _graph.Subgraph('sub')
        self.node = _graph.DuplicatedLeafNode('test', 'meta')

    def test_truthiness(self):
        self.assertTrue(self.node)

    def test_generate_asciitree_nodes(self):
        cache = {}
        graph_ids = {None: (f'%X{i}' for i in itertools.count()), self.subgraph: (f'%Y{i}' for i in itertools.count())}
        with self.subTest('first'):
            self.assertEqual(list(self.node._generate_asciitree_nodes(cache, graph_ids, 'S', 'B')), [
                             'Stest\n'])
        with self.subTest('second'):
            self.assertEqual(list(self.node._generate_asciitree_nodes(cache, graph_ids, 'S', 'B')), [
                             'Stest\n'])

    def test_collect_graphviz_nodes_edges(self):
        cache = {}
        nodes = {}
        edges = []
        cnt = map(str, itertools.count())
        with self.subTest('first-root'):
            self.assertEqual(self.node._collect_graphviz_nodes_edges(cache, cnt, nodes, edges, None, None), '0')
            self.assertEqual(edges, [])
            self.assertEqual(nodes, {None: ['0 [shape=box,label="test"];']})
        with self.subTest('second-root'):
            self.assertEqual(self.node._collect_graphviz_nodes_edges(cache, cnt, nodes, edges, None, None), '1')
            self.assertEqual(edges, [])
            self.assertEqual(nodes, {None: ['0 [shape=box,label="test"];', '1 [shape=box,label="test"];']})
        with self.subTest('sub'):
            self.assertEqual(self.node._collect_graphviz_nodes_edges(cache, cnt, nodes, edges, self.subgraph, None), '2')
            self.assertEqual(edges, [])
            self.assertEqual(nodes, {None: ['0 [shape=box,label="test"];', '1 [shape=box,label="test"];'], self.subgraph: ['2 [shape=box,label="test"];']})

    def test_graphviz_fill_color(self):
        with self.subTest('some'):
            nodes = {}
            self.node._collect_graphviz_nodes_edges({}, map(str, itertools.count()), nodes, [], None, lambda n: 'red')
            self.assertEqual(nodes[None][0], '0 [shape=box,label="test",style=filled,fillcolor="red"];')
        with self.subTest('none'):
            nodes = {}
            self.node._collect_graphviz_nodes_edges({}, map(str, itertools.count()), nodes, [], None, lambda n: None)
            self.assertEqual(nodes[None][0], '0 [shape=box,label="test"];')

    def test_walk(self):
        seen = set()
        self.assertEqual(list(self.node.walk(seen)), [self.node])
        self.assertEqual(list(self.node.walk(seen)), [])


class InvisibleNode(TestCase):

    def setUp(self):
        super().setUp()
        self.node = _graph.InvisibleNode('meta')

    def test_truthiness(self):
        self.assertFalse(self.node)

    def test_generate_asciitree_nodes(self):
        cache = {}
        self.assertEqual(list(self.node._generate_asciitree_nodes({}, {None: (f'%X{i}' for i in itertools.count())}, 'S', 'B')), ['S\n'])

    def test_collect_graphviz_nodes_edges(self):
        cache = {}
        nodes = {}
        edges = []
        cnt = map(str, itertools.count())
        self.assertEqual(self.node._collect_graphviz_nodes_edges({}, map(str, itertools.count()), nodes, edges, None, None), None)
        self.assertEqual(edges, [])
        self.assertEqual(nodes, {})

    def test_walk(self):
        seen = set()
        self.assertEqual(list(self.node.walk(seen)), [self.node])
        self.assertEqual(list(self.node.walk(seen)), [])


class generate(TestCase):

    def setUp(self):
        super().setUp()
        a = _graph.RegularNode('a', (), {}, None)
        b = _graph.RegularNode('b', (a,), {}, None)
        c = _graph.RegularNode('c', (a, b), {}, 'red')
        d = _graph.RegularNode('d', (c, a), {}, None)
        self.single = d
        B = _graph.Subgraph('B', None)
        C = _graph.Subgraph('C', B)
        D = _graph.Subgraph('D', B)
        E = _graph.Subgraph('E', C)
        e = _graph.RegularNode('e', (a,), {}, None, E)
        f = _graph.RegularNode('f', (b, e), {}, None, C)
        g = _graph.RegularNode('g', (a,), {}, None, D)
        h = _graph.RegularNode('h', (d, f), {}, None, B)
        i = _graph.RegularNode('i', (e, f, g), {}, None, E)
        j = _graph.RegularNode('j', (i,), {}, None)
        self.multiple = j

    def test_single_asciitree_rich(self):
        self.assertEqual(self.single.generate_asciitree(True),
                         '%0 = d\n'
                         '├ %1 = c\n'
                         '│ ├ %2 = a\n'
                         '│ └ %3 = b\n'
                         '│   └ %2\n'
                         '└ %2\n')

    def test_single_asciitree_unrich(self):
        self.assertEqual(self.single.generate_asciitree(False),
                         '%0 = d\n'
                         ': %1 = c\n'
                         '| : %2 = a\n'
                         '| : %3 = b\n'
                         '|   : %2\n'
                         ': %2\n')

    def test_single_graphviz_source(self):
        self.assertEqual(self.single.generate_graphviz_source(),
                         'digraph {'
                         'bgcolor="darkgray";'
                         '0 [shape=box,label="d"];'
                         '1 [shape=box,label="c"];'
                         '2 [shape=box,label="a"];'
                         '3 [shape=box,label="b"];'
                         '2 -> 1;'
                         '2 -> 3;'
                         '3 -> 1;'
                         '1 -> 0;'
                         '2 -> 0;'
                         '}')

    def test_single_graphviz_source_fill_color(self):
        self.assertEqual(self.single.generate_graphviz_source(fill_color=lambda node: node.metadata),
                         'digraph {'
                         'bgcolor="darkgray";'
                         '0 [shape=box,label="d"];'
                         '1 [shape=box,label="c",style=filled,fillcolor="red"];'
                         '2 [shape=box,label="a"];'
                         '3 [shape=box,label="b"];'
                         '2 -> 1;'
                         '2 -> 3;'
                         '3 -> 1;'
                         '1 -> 0;'
                         '2 -> 0;'
                         '}')

    def test_multiple_asciitree(self):
        self.assertEqual(self.multiple.generate_asciitree(True),
                         'SUBGRAPHS\n'
                         'A\n'
                         '└ B = B\n'
                         '  ├ C = C\n'
                         '  │ └ D = E\n'
                         '  └ E = D\n'
                         'NODES\n'
                         '%A0 = j\n'
                         '└ %D0 = i\n'
                         '  ├ %D1 = e\n'
                         '  │ └ %A1 = a\n'
                         '  ├ %C0 = f\n'
                         '  │ ├ %A2 = b\n'
                         '  │ │ └ %A1\n'
                         '  │ └ %D1\n'
                         '  └ %E0 = g\n'
                         '    └ %A1\n')

    def test_multiple_graphviz_source(self):
        self.assertEqual(self.multiple.generate_graphviz_source(),
                         'digraph {'
                         'bgcolor="darkgray";'
                         'subgraph cluster7 {'
                         'bgcolor="lightgray";'
                         'color="none";'
                         'subgraph cluster8 {'
                         'bgcolor="lightgray";'
                         'color="none";'
                         'subgraph cluster9 {'
                         'bgcolor="lightgray";'
                         'color="none";'
                         '1 [shape=box,label="i"];'
                         '2 [shape=box,label="e"];'
                         '}'
                         '4 [shape=box,label="f"];'
                         '}'
                         'subgraph cluster10 {'
                         'bgcolor="lightgray";'
                         'color="none";'
                         '6 [shape=box,label="g"];'
                         '}'
                         '}'
                         '0 [shape=box,label="j"];'
                         '3 [shape=box,label="a"];'
                         '5 [shape=box,label="b"];'
                         '3 -> 2;'
                         '2 -> 1;'
                         '3 -> 5;'
                         '5 -> 4;'
                         '2 -> 4;'
                         '4 -> 1;'
                         '3 -> 6;'
                         '6 -> 1;'
                         '1 -> 0;'
                         '}')

    def test_export_graphviz(self):
        try:
            self.multiple.export_graphviz()
        except FileNotFoundError:
            self.skipTest('graphviz not available')
