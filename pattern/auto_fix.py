import ast
import astunparse
import copy

# TODO: how to identify the original dataset?
# TODO: how to get all dependent lines of split?
# TODO: assumption: only need to deal with top level nodes

r = open('sample.py', 'r')
tree = ast.parse(r.read(), 'sample.py')

# tree.body is a list of ast objects: import, assign, expr, etc.
# for line in tree.body:
#     print(ast.dump(line))

# return if a node is a split
def is_split(node):
    # slicing: 1. Assign 2. value is Subscript 3. Subscript.value is the dataset
    # note: Assign.targets is a list
    if isinstance(node, ast.Assign) and isinstance(node.value, ast.Subscript):
        return True
    return False

# if node is original dataset, return its name
# if not, return None
def is_dataset(node):
    if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and\
        isinstance(node.value.func, ast.Attribute) and isinstance(node.value.func.value, ast.Name) and\
        node.value.func.value.id == 'pd' and node.value.func.attr == 'DataFrame':  # identify pd?
        return node.targets[0].id  # what if multiple targets
    return None

suspicious_op_list = ['fillna']  # TODO: add more to the list
# return if a node is a problematic transform
# dataset cannot be None
def is_suspicious_transform(node, dataset):
    # an expr with dataset subscript's attribute function: df[_].__()
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and\
        isinstance(node.value.func, ast.Attribute) and isinstance(node.value.func.value, ast.Subscript)\
        and node.value.func.value.value.id == dataset and node.value.func.attr in suspicious_op_list:
        return True
    # TODO: other situations
    return False

# traverse every node of the tree
dataset = None
for node in ast.walk(tree):
    if is_dataset(node):
        dataset = is_dataset(node)

index_record = {'split': [], 'transform': []}
index = 0
for node in tree.body:
    if is_split(node):
        index_record['split'].append(index)
    if dataset and is_suspicious_transform(node, dataset):
        index_record['transform'].append(index)
    index += 1

# modify the tree:
train_name = tree.body[index_record['split'][0]].targets[0].id  # TODO: how to make this more intelligent?
test_name = tree.body[index_record['split'][1]].targets[0].id
# print(train_name, test_name)
# first, insert the split node right before the first transform node (or first suspicious transform)
transform_start = min(index_record['transform'])
# cannot insert a list of nodes
split_body = [tree.body[i] for i in index_record['split']]  # TODO: copy?
inserted = 0
for node in split_body:
    tree.body.insert(transform_start + inserted, node)
    inserted += 1
removed = 0
for i in index_record['split']:
    # remove() only removes first occurence of value
    # tree.body.remove(tree.body[i + len(split_body) - removed])
    del tree.body[i + len(split_body) - removed]
    removed += 1

# now update and duplicate transforms on train and test dataset
new_transform_loc = [i + len(index_record['split']) for i in index_record['transform']]
transform_nodes = [tree.body[i] for i in new_transform_loc]
# print(ast.dump(transform_nodes[0]))


# TODO: create a new node that replaces all dataset variable to train dataset variable
class VariableReplacement(ast.NodeTransformer):
    def __init__(self, old_var, new_var):
        self.old_var = old_var
        self.new_var = new_var
    
    def visit(self, node):
        self.generic_visit(node)  # recursive, post-order
        if isinstance(node, ast.Name) and node.id == self.old_var:
            return ast.Name(self.new_var, node.ctx)
        return node

class ModifyTransform(ast.NodeTransformer):
    def __init__(self, dataset, train_name, test_name):
        self.dataset = dataset
        self.train_name = train_name
        self.test_name = test_name
    
    def visit_Expr(self, node):
        if is_suspicious_transform(node, self.dataset):
            # replace this node with two nodes
            train_node = copy.deepcopy(node)
            test_node = copy.deepcopy(node)
            # TODO: if the node is df[].call(...df...) type
            train_node.value.func.value.value.id = self.train_name
            test_node.value.func.value.value.id = self.test_name
            variable_replacer = VariableReplacement(self.dataset, self.train_name)
            train_node = variable_replacer.visit(train_node)
            test_node = variable_replacer.visit(test_node)
            
            return [train_node, test_node]  # ast.copy_location()
        return node

tree = ModifyTransform(dataset, train_name, test_name).visit(tree)
# unparse and output
print('\n', astunparse.unparse(tree))

# print('\n', astunparse.dump(tree))
