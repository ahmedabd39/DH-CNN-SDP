from dataStructures import Stack, Graph


class PythonParser(object):
    filename, lines = None, []

    def __init__(self, filename):
        super(PythonParser, self).__init__()
        self.filename = filename

    def construct_graph(self):
        nodes, edges = [], []
        node_info = [[]]
        indentation = []
        with open(self.filename, 'r') as file_obj:
            self.lines = file_obj.readlines()

        # The parser works like a pushdown automata
        # State 0: Basic block
        # State 1: Just encountered an if
        # State 2: Inside if or else block
        # State 3: Just encountered an else
        STATE, node_i = 0, 0
        line_no = 0
        stack = Stack()
        while line_no <= len(self.lines) - 1:
            line = self.lines[line_no]
            indentation.append((len(line) - len(line.lstrip(' ')))//4)

            if STATE == 0:
                if line.strip().startswith('if'):
                    STATE = 1
                    node_i += 1
                    node_info.append([])
                    stack.push(line_no+1)
                    edges.append((node_i, node_i+1))
                elif line.strip().startswith('while'):
                    STATE = 1
                    node_i += 1
                    node_info.append([])
                    stack.push(line_no+1)
                    edges.append((node_i, node_i+1))
                
            elif STATE == 1:
                STATE = 2
                node_i += 1
                node_info.append([])
                edges.append((node_i, node_i+1))
            elif STATE == 3:
                STATE = 2
                node_i += 1
                node_info.append([])
            elif STATE == 2:
                popped = []
                for d in range(indentation[line_no-1] - indentation[line_no]):
                    popped.append(stack.pop())
                if indentation[line_no-1] - indentation[line_no] > 0:
                    node_i += 1
                    node_info.append([])
                    edges.append((node_i, node_i+1))
                    for popped_item in popped:
                        for node in node_info:
                            if popped_item in node:
                                popped_node = node_info.index(node)+1
                                edges.append((popped_node, node_i+1))
                    if stack.isEmpty():
                        STATE = 0
                if line.strip().startswith('if'):
                    STATE = 1
                    node_i += 1
                    node_info.append([])
                    stack.push(line_no+1)
                    edges.append((node_i, node_i+1))
                elif line.strip().startswith('while'):
                    STATE = 1
                    node_i += 1
                    node_info.append([])
                    stack.push(line_no+1)
                    edges.append((node_i, node_i+1))
                #elif line.strip().startswith('else'):
                    #STATE = 3
                    #popped = stack.pop()
                    #line_no += 1
                    #continue

            node_info[node_i].append(line_no+1)
            print(line_no+1, STATE)
            line_no += 1

        if STATE == 2:
            node_i += 1
            node_info.append([])
            edges.append((node_i, node_i+1))
            while not stack.isEmpty():
                popped_item = stack.pop()
                for node in node_info:
                    if popped_item in node:
                        popped_node = node_info.index(node)+1
                        edges.append((popped_node, node_i+1))
            STATE = 0

        if STATE == 0:
            nodes = [i for i in range(1, len(node_info)+1)]
            return Graph(nodes, edges)
        else:
            print("Parsing unsuccessful at state", STATE)
