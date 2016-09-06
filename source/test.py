from abc import ABCMeta, abstractmethod, abstractproperty

class A(object):
    def __init__(self):
        pass
    @abstractmethod
    def f_a(self):
        print('in class A')

class B(A):
    def __init__(self):
        super(B, self).__init__()
        pass

    # @abstractmethod
    # def f_a(self):
    #     print('in class B')

class C(B):
    def __init__(self):
        super(C, self).__init__()
        pass

    def f_a(self):
        print('in class C')


class A:
    def __init__(self):
        self.one="one"
class B:
    def __init__(self):
        self.two="two"

class C(A,B):
    def __init__(self):

        A.__init__(self)
        B.__init__(self)
    def printselfnum(self):
        print(self.one,self.two)
c=C()
c.printselfnum()