#Linearly separable dataset
import matplotlib.pyplot as plt

c1=np.array([j for i,j in enumerate(data.X) if data.y[i]==0])
x1_c1=c1[:,0]
x2_c1=c1[:,1]

c2= np.array([j for i,j in enumerate(data.X) if data.y[i]==1])
x1_c2=c2[:,0]
x2_c2=c2[:,1]


plt.figure(1)
plt.title("Linear Nature of Data")
plt.xlabel('x1')
plt.ylabel('x2')
plt.scatter(x1_c1,x2_c1,color='b', label="C1")
plt.scatter(x1_c2,x2_c2,color='r', label='C2')
plt.legend()
