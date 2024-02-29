from pycparser import c_ast,c_parser,parse_file

# 定义一个名为 Tree 的类，代表一个树形结构
class AST_Tree(object):
    # 初始化方法，设置 parent 为 None，num_children 为 0，children 为一个空列表
    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.attribute = ''
        self.children = list()

    # 添加子节点的方法，将子节点的 parent 设置为自身，num_children 加 1，将子节点添加到 children 列表中
    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def print(self,prefix):
        print(prefix,self.attribute,'deg =',self.num_children)
        for child in self.children:
            child.print(prefix+'-')

    def get_all_attributes(self):
        attributes=[]
        attributes.append(self.attribute)
        for child in self.children:
            attributes.extend(child.get_all_attributes())
        return attributes

class ASTVisitor(c_ast.NodeVisitor):
    def getAttribute(self,attr):
        # print(attr)
        return attr

    def getAST(self,node,parent):
        tree=AST_Tree()
        tree.parent=parent
        nodeName = node.__class__.__name__
        nodeChildren = node.children()
        nodeAttributes = node.attr_names
        
        tree.attribute=nodeName
        # if nodeAttributes:
        #     for attr in nodeAttributes:
        #         tree.attribute.append(self.getAttribute(attr))
        for _, n in nodeChildren:
            son=self.getAST(n,tree)
            tree.add_child(son)

        return tree