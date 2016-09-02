class A(object):
    def __init__(self):
        pass

    def f_a(self):
        A.f_b()

    @classmethod
    def f_b(cls):
        print('word')

a = A()
a.f_a()