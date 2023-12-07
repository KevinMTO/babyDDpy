
class Addition():
    def __init__(self):
        self.complexNumber = 0. + 0.j

    def add(self, x, y):
        before = self.complexNumber.cacheCount()

        result = self.add2(x, y)

        if result.weight != Complex.zero:
            self.complexNumber.returnToCache(result.weight)
            result.weight = self.complexNumber.lookup(result.weight)

        after = self.complexNumber.complexCache.getCount()
        assert after == before

        return result

    def add2(self, x, y):
        if x.nextNode is None:
            return y
        if y.nextNode is None:
            return x

        if x.weight == Complex.zero:
            if y.weight == Complex.zero:
                return y
            result = y
            result.weight = self.complexNumber.getCached(x.weight.real, x.weight.img)
            return result

        if y.weight == Complex.zero:
            result = x
            result.weight = self.complexNumber.getCached(y.weight.real, y.weight.img)
            return result

        if x.nextNode == y.nextNode:
            result = y
            result.weight = self.complexNumber.addCached(x.weight, y.weight)
            if result.weight.approximatelyZero():
                self.complexNumber.returnToCache(result.weight)
                return Edge.zero()

            return result

        compute_table = self.getAddComputeTable()
        result = compute_table.lookup((x.nextNode, x.weight), (y.nextNode, y.weight))

        if result.nextNode is not None:
            if result.weight.approximatelyZero():
                return Edge.zero()

            return Edge(result.nextNode, self.complexNumber.getCached(result.weight))

        new_successor = 0

        if x.is_terminal():
            new_successor = y.nextNode.varIndx
        else:
            new_successor = x.nextNode.varIndx
            if not y.is_terminal() and y.nextNode.varIndx > new_successor:
                new_successor = y.nextNode.varIndx

        edge_sum = [Edge.zero() for _ in range(len(x.nextNode.edges))]

        for i in range(len(x.nextNode.edges)):
            e1 = Edge()

            if not x.is_terminal() and x.nextNode.varIndx == new_successor:
                e1 = x.nextNode.edges[i]

                if e1.weight != Complex.zero:
                    e1.weight = self.complexNumber.mulCached(e1.weight, x.weight)
            else:
                e1 = x
                if y.nextNode.edges[i].nextNode is None:
                    e1 = Edge(None, Complex.zero)

            e2 = Edge()
            if not y.is_terminal() and y.nextNode.varIndx == new_successor:
                e2 = y.nextNode.edges[i]

                if e2.weight != Complex.zero:
                    e2.weight = self.complexNumber.mulCached(e2.weight, y.weight)
            else:
                e2 = y
                if x.nextNode.edges[i].nextNode is None:
                    e2 = Edge(None, Complex.zero)

            edge_sum[i] = self.add2(e1, e2)

            if not x.is_terminal() and x.nextNode.varIndx == new_successor and e1.weight != Complex.zero:
                self.complexNumber.returnToCache(e1.weight)

            if not y.is_terminal() and y.nextNode.varIndx == new_successor and e2.weight != Complex.zero:
                self.complexNumber.returnToCache(e2.weight)

        e = makeDDNode(new_successor, edge_sum, True)
        compute_table.insert((x.nextNode, x.weight), (y.nextNode, y.weight), (e.nextNode, e.weight))
        return e
