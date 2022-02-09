from math import factorial
from scipy.linalg import block_diag
import numpy as np
from qpsolvers import solve_qp
import matplotlib.pyplot as plt

class min_snap:
    def __init__(self,velocity, max_velocity,  parameters):
        self.num_points = len(parameters) 
        self.k = self.num_points - 1
        self.n = 7 
        self.get_points(parameters)
        self.velocity = velocity
        self.parameters = parameters
        self.time_stamps = self.get_time_stamps(self.velocity) 
        self.b_x , self.b_y , self.b_z = self.get_b()
        self.q = np.zeros(((self.n+1) * self.k, 1)).reshape(((self.n+1)*self.k,))
        self.G = np.zeros((4 * self.k + 2, (self.n+1) * self.k))
        self.h = np.zeros((4 * self.k + 2, 1)).reshape((4 * self.k + 2, ))
        self.max_velocity = max_velocity 
        self.minimum_time_stamps = self.get_time_stamps(self.max_velocity) 
        self.t_segments = self.get_time_segments(self.time_stamps)
        self.minimum_time_segments = self.get_time_segments(self.minimum_time_stamps)
        self.get_Q()
        self.get_A()

    def get_points(self,parameters):
        x = []
        y = []
        z = []
        for point in parameters :
            x.append(point[0])
            y.append(point[1])
            z.append(point[2])
        self.x = x
        self.y = y
        self.z = z
    
    def get_time_stamps(self, velocity):
        t = [0.1] 
        parameters = self.parameters
        for i in range(self.k):
            current_point = np.array(parameters[i])
            next_point = np.array(parameters[i+1])
            distance = np.linalg.norm(current_point - next_point)
            time_stamp = distance / velocity
            t.append(t[-1] + time_stamp) 
        return t

    def get_time_segments(self, time_stamps):
        t = time_stamps
        time_segments = []
        for i in range(1,self.k + 1):
            time_segments.append(t[i] - t[i - 1])
        return time_segments

    def get_Q(self): 
        Q_s = []
        n = self.n + 1
        t = self.time_stamps
        for l in range (1,self.k+1):
            Q = np.zeros((n,n))
            for i in range(n):
                for j in range (n):
                    if i > 3 and j > 3 :
                        Q[i][j] = (factorial(i)*factorial(j)*(t[l]**(i + j - 7) - t[l-1]**(i + j - 7)))/(factorial(i-4) * factorial(j-4) * (i + j - 7))
            Q_s.append(Q)
        to_attach = Q_s[0]
        Q = []
        for i in range(len(Q_s) - 1):
            Q = block_diag(to_attach,Q_s[i+1])
            to_attach = Q
        Q = Q + (0.0001 * np.identity((self.k*n))) # adding the term to ensure that Q is a positive definite matrix
        self.Q = Q

    def get_A(self):
        k = self.k
        n = self.n
        A = np.zeros((4*k + 2 , k * (n + 1)))
        t = self.time_stamps
        for j in range (n + 1):
            A[0][j] = t[0]**j
            A[1][j] = j*t[0]**(j-1)
            A[2][j] = j*(j-1)*t[0]**(j-2)

        for i in range (1, k) :
            for l in range (n+1):
                A[i + 2][(i-1) * (n + 1) + l] = t[i]**l

        for j in range ((k - 1) * (n + 1), k * (n + 1)):
            r = j - (k - 1) * (n + 1)
            A[k+2][j] = t[k]**(r)
            A[k+3][j] = r*t[k]**(r-1)
            A[k+4][j] = r*(r-1)*t[k]**(r-2)

        for i in range(k-1):
            for l in range (2*n + 2):
                if l < (n + 1):
                    A[k + 5 + 3 * i][(n+1)*i + l] = t[i + 1]**l
                    A[k + 6 + 3 * i][(n+1)*i + l] = l*t[i + 1]**(l-1)
                    A[k + 7 + 3 * i][(n+1)*i + l] = l*(l-1)*t[i + 1]**(l-2)
                else:
                    A[k + 5 + 3 * i][(n+1)*i + l] = -t[i + 1]**(l-(n+1)) 
                    A[k + 6 + 3 * i][(n+1)*i + l] = -(l-(n+1))*t[i + 1]**((l-(n+1))-1)
                    A[k + 7 + 3 * i][(n+1)*i + l] = -(l-(n+1))*((l-(n+1))-1)*t[i + 1] ** ((l-(n+1))-2)
        self.A = A

    def get_b(self):
        points = [self.x,self.y,self.z]
        b = []
        for i in range(3):
            bi = np.array([points[i][0], 0.0, 0.0])
            last_point = np.array([points[i][self.num_points-1], 0.0, 0.0])
            bi = np.append(bi, points[i][1:(self.num_points-1)])
            bi = np.append(bi, last_point)
            bi = np.append(bi,np.zeros((3*(self.k - 1))))
            b.append(bi)
        return b

    def solve(self):
        self.p_x = solve_qp(self.Q,self.q,self.G,self.h,self.A,self.b_x)
        self.p_y = solve_qp(self.Q,self.q,self.G,self.h,self.A,self.b_y)
        self.p_z = solve_qp(self.Q,self.q,self.G,self.h,self.A,self.b_z)

    def optimize(self): 
        self.gradient_descent()

    def get_cost_function(self, k_T = 10000000):
        self.get_Q()
        self.get_A()        
        t_seg = np.copy(self.t_segments) 
        t = [0.1]
        for i in range (len(t_seg)):
            t.append(t[-1] + t_seg[i]) 
        self.time_stamps = np.copy(t)  
        self.solve()
        p_x, p_y, p_z = self.p_x, self.p_y, self.p_z
        Q = self.Q
        t = np.copy(self.time_stamps)
        time_term = k_T * (t[-1]- t[0])    
        x_term = 0.00001 * np.matmul(np.array(p_x).T,np.matmul(Q,np.array(p_x)))
        y_term = 0.00001 * np.matmul(np.array(p_y).T,np.matmul(Q,np.array(p_y)))
        z_term = 0.00001 * np.matmul(np.array(p_z).T,np.matmul(Q,np.array(p_z)))
        J = x_term + y_term + z_term + time_term
        return J / 100000


    def get_gradient(self):
        t_test_segments = np.copy(self.t_segments)
        t_segments_initial = np.copy(self.t_segments)
        gradient = np.zeros(len(self.t_segments)) 
        h = 0.0001
        J_prev = self.get_cost_function()
        for i in range (len(self.t_segments)):
            if (t_test_segments[i] < self.minimum_time_segments[i]): 
                gradient[i] = 0 
            else :
                t_test_segments[i] = t_test_segments[i] + h 
                self.t_segments = np.copy(t_test_segments) 
                J_curr = self.get_cost_function() 
                gradient[i] = (J_curr - J_prev)/h 
                t_test_segments[i] = t_test_segments[i] - h 
        self.t_segments = np.copy(t_segments_initial)
        return gradient
    


    def gradient_descent(self, num_iterations = 1000, descent_rate = 0.00001, threshold = 0.005):
        print ("Total Time before time scaling:", self.time_stamps[-1] - self.time_stamps[0])
        iterator = 0
        optimize_time_segments = np.copy(self.t_segments)
        difference = 10000000000
        while (difference > threshold) and  (iterator < num_iterations):
            prev_cost = self.get_cost_function()
            gradient = self.get_gradient()
            delta_t = descent_rate * gradient
            optimize_time_segments = optimize_time_segments - delta_t
            self.t_segments = np.copy(optimize_time_segments)
            current_cost = self.get_cost_function()
            difference = abs(current_cost - prev_cost)
            iterator = iterator + 1

        print ("Number of iterations of gradient descent:", iterator)
        print ("Total time after time scaling: ", self.time_stamps[-1] - self.time_stamps[0])

    
    def plot(self, time_resolution):
        plt.figure()
        ax = plt.axes(projection = '3d')
        ax.scatter(self.x, self.y, self.z, 'b', marker= 'o')
        self.solve()
        p_x, p_y, p_z = self.p_x, self.p_y, self.p_z
        for i in range (self.k):
            x_segment = []
            y_segment = []
            z_segment = []
            t = np.linspace(self.time_stamps[i],self.time_stamps[i + 1], time_resolution)
            for j in range (time_resolution):
                x_term, y_term, z_term = 0, 0, 0
                for l in range ((self.n+1)*i, (self.n+1)*(i+1)):
                    x_term = x_term + p_x[l]*(t[j]**(l-(self.n+1)*i))
                    y_term = y_term + p_y[l]*(t[j]**(l-(self.n+1)*i))
                    z_term = z_term + p_z[l]*(t[j]**(l-(self.n+1)*i))
                x_segment.append(x_term)
                y_segment.append(y_term)
                z_segment.append(z_term)
            ax.plot3D(x_segment,y_segment,z_segment, 'r')
        plt.show()


if __name__ == '__main__':
    print ("Input format: ")
    print("x-coordinate y-coordinate z-coordinate")
    n = int(input("Enter the number of points: "))
    points = []
    for i in range(n):
        point = input(f"Enter point number {i+1}: ").split()
        x = float(point[0])
        y = float(point[1])
        z = float(point[2])
        points.append((x,y,z))
    
    v = float(input("Enter the average velocity to be followed: "))
    v_max = float(input("Enter the maximum velocity that can be followed: "))
    minsnap = min_snap(v, v_max, points)
    minsnap.optimize()
    minsnap.plot(100)