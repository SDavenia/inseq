class A:
    def encode(self):
        return "Encoding in A"

class X:
    pass

class Y:
    def encode(self):
        return "Encoding in Y"

class AX(A, X):
    # Inherits the encode method from A
    pass

class AY(Y, A):
    # Inherits the encode method from Y due to method resolution order (MRO)
    pass

# Test the classes
ax_instance = AX()
ay_instance = AY()

print(ax_instance.encode())  # Should print "Encoding in A"
print(ay_instance.encode())  # Should print "Encoding in Y"