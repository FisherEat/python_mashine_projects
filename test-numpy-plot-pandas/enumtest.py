from enum import Enum

Month = Enum('Month', ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'))

for name, member in Month.__members__.items():
    print(name, '=>', member, ',', member.value)

def fn(self, name = 'world-----'):
    print('Hello, %s.' % name)

Hello = type('Hello', (object,), dict(hello=fn))

h = Hello()

print(h.hello())


def foo():
    r = some_function()
    if r == (-1):
        return (-1)
    return r

def bar():
    r = foo()
    if r == (-1):
        print('Error')
    else:
        pass

try:
    print('try...')
    r = 10/int('a')
    print('result:',r)
except ValueError as e:
    print('ValueError:', e)
except ZeroDivisionError as e:
    print('ZeroDivisionError:',e)
finally:
    print('finnally...')
print('END')


