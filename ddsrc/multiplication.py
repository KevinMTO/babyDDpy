

class Multiplication():
    def __init__(self):
        self.complexNumber = 0 + 0.j

    def multiply(self, x, y):
        before = self.complexNumber.cacheCount()

        var = -1

        if not x.is_terminal():
            var = x.nextNode.varIndx
        if not y.is_terminal() and y.nextNode.varIndx > var:
            var = y.nextNode.varIndx

        e = self.multiply2(x, y, var, start)

        if e.weight != Complex.zero and e.weight != Complex.one:
            self.complexNumber.returnToCache(e.weight)
            e.weight = self.complexNumber.lookup(e.weight)

        after = self.complexNumber.cacheCount()
        assert before == after

        return e

    def multiply2(self, x, y, var, start=QuantumRegister(0)):
        LEdge = Edge[LeftOperand]
        REdge = Edge[RightOperand]
        ResultEdge = Edge[RightOperand]

        if x.nextNode is None:
            return ResultEdge.zero()
        if y.nextNode is None:
            return y

        if x.weight == Complex.zero or y.weight == Complex.zero:
            return ResultEdge.zero()

        if var == start.value - 1:
            return ResultEdge.terminal(self.complexNumber.mulCached(x.weight, y.weight))

        x_copy = LEdge(next_node=x.nextNode, weight=Complex.one)
        y_copy = REdge(next_node=y.nextNode, weight=Complex.one)

        compute_table = self.get_multiplication_compute_table()
        lookup_result = compute_table.lookup(x_copy, y_copy)

        if lookup_result.nextNode is not None:
            if lookup_result.weight.approximatelyZero():
                return ResultEdge.zero()

            res_edge_init = ResultEdge(next_node=lookup_result.nextNode, weight=self.complexNumber.getCached(lookup_result.weight))

            self.complexNumber.mul(res_edge_init.weight, res_edge_init.weight, x.weight)
            self.complexNumber.mul(res_edge_init.weight, res_edge_init.weight, y.weight)

            if res_edge_init.weight.approximatelyZero():
                self.complexNumber.returnToCache(res_edge_init.weight)
                return ResultEdge.zero()

            return res_edge_init

        result_edge = ResultEdge()

        if x.nextNode.varIndx == var and x.nextNode.varIndx == y.nextNode.varIndx:
            if x.nextNode.identity:
                if isinstance(y.nextNode, mNode):
                    if y.nextNode.identity:
                        result_edge = makeIdent(start, var)
                    else:
                        result_edge = y_copy
                else:
                    result_edge = y_copy

                compute_table.insert(x_copy, y_copy, Edge(next_node=result_edge.nextNode, weight=result_edge.weight))
                result_edge.weight = self.complexNumber.mulCached(x.weight, y.weight)

                if result_edge.weight.approximatelyZero():
                    self.complexNumber.returnToCache(result_edge.weight)
                    return ResultEdge.zero()

                return result_edge

            if isinstance(y.nextNode, mNode):
                if y.nextNode.identity:
                    result_edge = x_copy
                    compute_table.insert(x_copy, y_copy, Edge(next_node=result_edge.nextNode, weight=result_edge.weight))
                    result_edge.weight = self.complexNumber.mulCached(x.weight, y.weight)

                    if result_edge.weight.approximatelyZero():
                        self.complexNumber.returnToCache(result_edge.weight)
                        return ResultEdge.zero()

                    return result_edge

        rows = 1 if x.is_terminal() else registers_sizes[x.nextNode.varIndx]
        cols = 1 if isinstance(y.nextNode, mNode) else registers_sizes[y.nextNode.varIndx] if not y.is_terminal() else 1
        multiplication_boundary = 1 if x.is_terminal() else registers_sizes[x.nextNode.varIndx]

        edge = [ResultEdge.zero() for _ in range(multiplication_boundary * cols)]

        for i in range(rows):
            for j in range(cols):
                idx = cols * i + j

                for k in range(multiplication_boundary):
                    e1 = LEdge() if x.is_terminal() or x.nextNode.varIndx != var else x.nextNode.edges[rows * i + k]
                    e2 = REdge() if y.is_terminal() or y.nextNode.varIndx != var else y.nextNode.edges[j + cols * k]

                    multiplied_recur_res = self.multiply2(e1, e2, QuantumRegister(var - 1), start)

                    if k == 0 or edge[idx].weight == Complex.zero:
                        edge[idx] = multiplied_recur_res
                    elif multiplied_recur_res.weight != Complex.zero:
                        old_edge = edge[idx]
                        edge[idx] = self.add2(old_edge, multiplied_recur_res)
                        self.complexNumber.returnToCache(old_edge.weight)
                        self.complexNumber.returnToCache(multiplied_recur_res.weight)

        result_edge = makeDDNode(var, edge, True)

        compute_table.insert(x_copy, y_copy, Edge(next_node=result_edge.nextNode, weight=result_edge.weight))

        if result_edge.weight != Complex.zero and (x.weight != Complex.one or y.weight != Complex.one):
            if result_edge.weight == Complex.one:
                result_edge.weight = self.complexNumber.mulCached(x.weight, y.weight)
            else:
                self.complexNumber.mul(result_edge.weight, result_edge.weight, x.weight)
                self.complexNumber.mul(result_edge.weight, result_edge.weight, y.weight)

            if result_edge.weight.approximatelyZero():
                self.complexNumber.returnToCache(result_edge.weight)
                return ResultEdge.zero()

        return result_edge
