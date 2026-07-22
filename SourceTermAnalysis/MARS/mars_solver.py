import numpy as np

class MarsSolver:
    '''
        对方程离散求解
    '''
    @classmethod
    def _thomas(cls,diagonal1,diagonal2,diagonal3,vector):
        '''
            追赶法求解三对角线方程组
        '''
        x = np.zeros_like(vector)
        y = np.zeros_like(vector)
        u = np.zeros_like(vector)
        l = np.zeros_like(diagonal1)
       
        u[0] = diagonal2[0]
        for i in range(0, diagonal1.size):
            l[i] = diagonal1[i]/u[i]
            u[i+1] = diagonal2[i+1] - l[i]*diagonal3[i]
        y[0] = vector[0]
        for i in range(0,l.size):
            y[i+1] = vector[i+1] - l[i]*y[i]
        x[-1] = y[-1]/u[-1]
        for i in range(diagonal3.size - 1, -1, -1):
            x[i] = (y[i] - diagonal3[i]*x[i+1])/u[i]
        return x
    
    @classmethod
    def _penta_thomas(cls,diagonal1,diagonal2,diagonal3,diagonal4,diagonal5,vector):
        '''
            拓展的追赶法求解五对角线方程组
        '''
        a = diagonal1
        b = diagonal2
        c = diagonal3
        d = diagonal4
        e = diagonal5
        f = vector

        s = a
        m = [b[0]]
        l = [c[0]]
        p = [d[0]/c[0]]
        l.append(c[1]-m[0]*p[0])
        q = [e[0]/l[0]]

        n = len(c)
        for i in range(1,n-1):
            m.append(b[i] - s[i-1]*p[i-1])
            p.append((d[i]-m[i-1]*q[i-1])/l[i])
            if i != n-2:
                q.append(e[i]/l[i])
            l.append(c[i+1]-s[i-1]*q[i-1]-m[i]*p[i])

        m.append(b[n-2]-s[n-3]*p[n-3])
        l.append(c[n-1]-s[n-3]*q[n-3]-m[n-2]*p[n-2])
        p.append((d[n-2]-m[n-3]*q[n-3])/l[n-2])

        y = [f[0]/l[0]]
        y.append((f[1]-m[0]*y[0])/l[1])
        for i in range(2,n):
            y.append((f[i]-s[i-2]*y[i-2]-m[i-1]*y[i-1])/l[i])
        
        x = np.zeros_like(y)
        x[-1] = y[-1]
        x[-2] = y[-2] - p[-2]*x[-1]
        for i in range(n-3, -1, -1):
            x[i] = y[i] - p[i]*x[i+1] - q[i]*x[i+2]
            
        return x
    
    @classmethod
    def _fvm_euler_thomas(cls,x_env,time_step,decay_constant,boundary_transport_coef,x_k,D,r,volume,Q,w,area,gamma):
        '''
            空间离散: 有限体积
            时间离散: 隐式欧拉
            求解器: 追赶法
            x_env: 一类边界值
            time_step: 时间步长
            decay_constant: 一阶化学反应系数
            boundary_transport_coef: 二类边界值
            x_k: 初始条件
            D: 扩散系数
            N: 总网格数
            r: 数值层半径
            volume: 网格体积
            area: 数值层面积
            gamma: 有限体积法中间参数
            Q: 数值层单位体积产生率
        '''
        N = volume.size
        diagonal2 = (gamma[0])*(1/time_step + decay_constant) + 2*D[0]*area[1]/r[1]/volume[0]
        diagonal3 = (1-gamma[0])*(1/time_step+decay_constant) - 2*D[0]*area[1]/volume[0]/r[1]
        vector = (1-gamma[0])*x_k[1]/time_step + (gamma[0])/time_step*x_k[0] + Q[0]

        diagonal1 = -D[0:N-1]*area[1:N]/(-r[0:N-1]+r[1:N])/volume[1:N]
        diagonal2 = np.append(diagonal2, (gamma[1:N])*(1/time_step + decay_constant) +  D[0:N-1]*area[1:N]/(-r[0:N-1]+r[1:N])/volume[1:N] + D[1:N]*area[2:N+1]/(-r[1:N]+r[2:N+1])/volume[1:N])
        diagonal3 = np.append(diagonal3, (1-gamma[1:N])*(1/time_step+decay_constant) - D[1:N]*area[2:N+1]/volume[1:N]/(-r[1:N]+r[2:N+1]))
        vector = np.append(vector, (1-gamma[1:N])*x_k[2:N+1]/time_step + (gamma[1:N])*x_k[1:N]/time_step + Q[1:N])

        if boundary_transport_coef is not None:
            diagonal1 = np.append(diagonal1,D[N - 1] / (r[N] - r[N-1]))
            diagonal2 = np.append(diagonal2,-D[N - 1] / (r[N] - r[N-1]) - boundary_transport_coef)
            vector = np.append(vector,-boundary_transport_coef*x_env)
        else:
            diagonal1 = np.append(diagonal1,0.0)
            diagonal2 = np.append(diagonal2,1)
            vector = np.append(vector,x_env)

        x_next = cls._thomas(diagonal1,diagonal2,diagonal3,vector)
        return ((x_next[0:-1]+x_next[1:])/2,x_next)
    
    @classmethod
    def _fvm_trapezium_thomas(cls,x_env,time_step,decay_constant,boundary_transport_coef,x_k,D,r,volume,Q,w,area,gamma):
        '''
            空间离散: 有限体积
            时间离散: Crank-Nicolson
            求解器: 追赶法
            x_env: 一类边界值
            time_step: 时间步长
            decay_constant: 一阶化学反应系数
            boundary_transport_coef: 二类边界值
            x_k: 初始条件
            D: 扩散系数
            N: 总网格数
            r: 数值层半径
            volume: 网格体积
            area: 数值层面积
            gamma: 有限体积法中间参数
            Q: 数值层单位体积产生率
            w: C-N格式前后时间层的分配系数
        '''
        N = volume.size
        diagonal2 = gamma[0]/time_step + decay_constant*gamma[0]*w + D[0]*area[1]*w/r[1]/volume[0]
        diagonal3 = (1-gamma[0])/time_step + decay_constant*(1-gamma[0])*w - D[0]*area[1]*w/r[1]/volume[0]
        vector = (gamma[0]/time_step - decay_constant*gamma[0]*(1-w) - D[0]*area[1]*(1-w)/r[1]/volume[0])*x_k[0]
        vector += ((1-gamma[0])/time_step - decay_constant*(1-gamma[0])*(1-w) + D[0]*area[1]*(1-w)/r[1]/volume[0])*x_k[1]
        vector += Q[0]

        diagonal1 = -D[0:N-1]*area[1:N]*w/(r[1:N]-r[0:N-1])/volume[1:N]
        diagonal2 = np.append(diagonal2,( gamma[1:N]/time_step + decay_constant*gamma[1:N]*w + D[0:N-1]*area[1:N]*w/(r[1:N]-r[0:N-1])/volume[1:N] + D[1:N]*area[2:N+1]*w/(r[2:N+1]-r[1:N])/volume[1:N]))
        diagonal3 = np.append(diagonal3,( (1-gamma[1:N])/time_step +decay_constant*(1-gamma[1:N])*w -D[1:N]*area[2:N+1]*w/(r[2:N+1]-r[1:N])/volume[1:N] ))
        
        dmask = D[0:N-1]/(r[1:N]-r[0:N-1])*(1-w)*area[1:N]/volume[1:N]*x_k[0:N-1]
        dmask += (gamma[1:N]/time_step -decay_constant*gamma[1:N]*(1-w) -D[0:N-1]*area[1:N]*(1-w)/(r[1:N]-r[0:N-1])/volume[1:N] - D[1:N]*area[2:N+1]*(1-w)/(r[2:N+1]-r[1:N])/volume[1:N] )*x_k[1:N]
        dmask += ( (1-gamma[1:N])/time_step -decay_constant*(1-gamma[1:N])*(1-w) + D[1:N]*area[2:N+1]*(1-w)/(r[2:N+1]-r[1:N])/volume[1:N] )*x_k[2:N+1]
        dmask += Q[1:N]
        vector = np.append(vector,dmask)

        diagonal1 = np.append(diagonal1, D[N-1]/(r[N]-r[N-1]))
        diagonal2 = np.append(diagonal2,-D[N-1]/(r[N]-r[N-1]) - boundary_transport_coef)
        vector = np.append(vector, -boundary_transport_coef*x_env)
     
        x_next = cls._thomas(diagonal1,diagonal2,diagonal3,vector)
        return ((x_next[0:-1]+x_next[1:])/2,x_next)
    
    @classmethod
    def _fdmleapfrog_euler_thomas(cls,x_env,time_step,decay_constant,boundary_transport_coef,x_k,D,r,volume,Q,w,area,gamma):
        '''
            空间离散: 中央有限差分(跳蛙格式)
            时间离散: 隐式欧拉
            求解器: 追赶法
            x_env: 一类边界值
            time_step: 时间步长
            decay_constant: 一阶化学反应系数
            boundary_transport_coef: 二类边界值
            x_k: 初始条件
            D: 扩散系数
            N: 总网格数
            r: 数值层半径
            volume: 网格体积
            area: 数值层面积
            gamma: 有限体积法中间参数
            Q: 数值层单位体积产生率
        '''
        N = volume.size
        h = np.array(r[1:] - r[0:-1])
        h = np.append(h,h[-1])
        D_right = np.append((D[0:N-1]*h[1:N] + D[1:N]*h[0:N-1])/(h[1:N]+h[0:N-1]),0)
        D_left = np.append(0,(D[0:N-1]*h[1:N]+D[1:N]*h[0:N-1])/( h[0:N-1]+h[1:N]))
        
        k = 4*D[0]*h[1]/r[1]**2/r[2] - 4*D[0]/r[1]**2 -2*D_right[0]/h[0]/(h[0]+h[1]) - decay_constant
        l = 4*D[0]*h[0]/r[1]**2/r[2] + 2*D_right[0]/h[0]/(h[0]+h[1])
        diagonal2 = 1/time_step -k
        diagonal3 = -l
        vector = 1/time_step*x_k[0] + Q[0]

        e = -4*D[1:N]/(r[2:N+1]+r[1:N])/(h[1:N]+h[0:N-1]) + 2*D_left[1:N]/h[1:N]/(h[1:N]+h[0:N-1])
        f = 4*D[1:N]*h[2:N+1]/(r[2:N+1]+r[1:N])/h[1:N]/(h[2:N+1]+h[1:N]) -4*D[1:N]*h[0:N-1]/(r[2:N+1]+r[1:N])/h[1:N]/(h[0:N-1]+h[1:N]) -2*D_right[1:N]/h[1:N]/(h[1:N]+h[2:N+1]) - 2*D_left[1:N]/h[1:N]/(h[1:N]+h[0:N-1]) - decay_constant
        g = 4*D[1:N-1]/(r[2:N]+r[1:N-1])/(h[2:N]+h[1:N-1]) + 2*D_right[1:N-1]/h[1:N-1]/(h[2:N]+h[1:N-1])
        diagonal1 = -e[0:-1]
        diagonal2 = np.append(diagonal2,(1/time_step-f[0:-1]))
        diagonal3 = np.append(diagonal3,-g)
        vector = np.append(vector,1/time_step*x_k[1:N-1] + Q[1:N-1])

        add = D[N-1]/h[N-1] + boundary_transport_coef/2
        sub = D[N-1]/h[N-1] - boundary_transport_coef/2
        diagonal1 = np.append(diagonal1,-e[-1])
        q = ( 2*D[N-1]/(r[N]+r[N-1])/h[N-1] + D[N-1]/h[N-1]**2)
        diagonal2 = np.append(diagonal2,1/time_step - f[-1] - q*sub/add )
        vector = np.append(vector,x_k[N-1]/time_step + q*boundary_transport_coef*x_env/add + Q[N-1] ) 
        
        x_next = cls._thomas(diagonal1,diagonal2,diagonal3,vector)
        h = h[0:-1]
        x_image = sub/add*x_next[-1] + boundary_transport_coef*x_env/add
        x_nod = np.append(x_next[0] ,(x_next[0:-1]*h[1:]+x_next[1:]*h[0:-1])/(h[1:]+h[0:-1]))
        x_nod = np.append(x_nod,(x_image+x_next[-1])/4)
        return (x_next,x_nod)
    
    @classmethod
    def _fdmleapfrog_trapezium_thomas(cls,x_env,time_step,decay_constant,boundary_transport_coef,x_k,D,r,volume,Q,w,area,gamma):
        '''
            空间离散: 中央有限差分(跳蛙格式)
            时间离散: Crank-Nicolson
            求解器: 追赶法
            x_env: 一类边界值
            time_step: 时间步长
            decay_constant: 一阶化学反应系数
            boundary_transport_coef: 二类边界值
            x_k: 初始条件
            D: 扩散系数
            N: 总网格数
            r: 数值层半径
            volume: 网格体积
            area: 数值层面积
            gamma: 有限体积法中间参数
            Q: 数值层单位体积产生率
            w: C-N格式前后时间层的分配系数
        '''
        N = volume.size
        h = np.array(r[1:] - r[0:-1])
        h = np.append(h,h[-1])
        D_right = np.append((D[0:N-1]*h[1:N] + D[1:N]*h[0:N-1])/(h[1:N]+h[0:N-1]),0)
        D_left = np.append(0,(D[0:N-1]*h[1:N]+D[1:N]*h[0:N-1])/( h[0:N-1]+h[1:N]))

        k = 4*D[0]*h[1]/r[1]**2/r[2] - 4*D[0]/r[1]**2 -2*D_right[0]/h[0]/(h[0]+h[1]) - decay_constant
        l = 4*D[0]*h[0]/r[1]**2/r[2] + 2*D_right[0]/h[0]/(h[0]+h[1])
        diagonal2 = 1/time_step -k*w
        diagonal3 = -l*w
        vector = 1/time_step*x_k[0] + Q[0] + (1-w)*k*x_k[0] + (1-w)*l*x_k[1]

        e = -4*D[1:N]/(r[2:N+1]+r[1:N])/(h[1:N]+h[0:N-1]) + 2*D_left[1:N]/h[1:N]/(h[1:N]+h[0:N-1])
        f = 4*D[1:N]*h[2:N+1]/(r[2:N+1]+r[1:N])/h[1:N]/(h[2:N+1]+h[1:N]) -4*D[1:N]*h[0:N-1]/(r[2:N+1]+r[1:N])/h[1:N]/(h[0:N-1]+h[1:N]) -2*D_right[1:N]/h[1:N]/(h[1:N]+h[2:N+1]) - 2*D_left[1:N]/h[1:N]/(h[1:N]+h[0:N-1]) - decay_constant
        g = 4*D[1:N-1]/(r[2:N]+r[1:N-1])/(h[2:N]+h[1:N-1]) + 2*D_right[1:N-1]/h[1:N-1]/(h[2:N]+h[1:N-1])
        diagonal1 = -e[0:-1]*w
        diagonal2 = np.append(diagonal2,(1/time_step-f[0:-1]*w))
        diagonal3 = np.append(diagonal3,-g*w)
        vector = np.append(vector,1/time_step*x_k[1:N-1] + Q[1:N-1] + (1-w)*e[0:-1]*x_k[0:N-2] + (1-w)*f[0:-1]*x_k[1:N-1] + (1-w)*g*x_k[2:N])

        add = D[N-1]/h[N-1] + boundary_transport_coef/2
        sub = D[N-1]/h[N-1] - boundary_transport_coef/2
        q = ( 2*D[N-1]/(r[N]+r[N-1])/h[N-1] + D[N-1]/h[N-1]**2)
        diagonal1 = np.append(diagonal1,-e[-1]*w)
        diagonal2 = np.append(diagonal2,1/time_step - w*f[-1] - w*q*sub/add )
        vector = np.append(vector,x_k[N-1]/time_step + q*boundary_transport_coef*x_env/add + Q[N-1] + (1-w)*e[-1]*x_k[N-2] + (1-w)*(f[-1]+q*sub/add)*x_k[N-1] ) 

        x_next = cls._thomas(diagonal1,diagonal2,diagonal3,vector)
        h = h[0:-1]
        x_image = sub/add*x_next[-1] + boundary_transport_coef*x_env/add if boundary_transport_coef is not None else x_env
        x_nod = np.append(x_next[0] ,(x_next[0:-1]*h[1:]+x_next[1:]*h[0:-1])/(h[1:]+h[0:-1]))
        x_nod = np.append(x_nod,(x_image+x_next[-1])/4)
        return (x_next,x_nod)
    
    @classmethod
    def _fdmtaylor_euler_thomas(cls,x_env,time_step,decay_constant,boundary_transport_coef,x_k,D,r,volume,Q,w,area,gamma):
        '''
            空间离散: 高阶有限差分(五点四阶格式)
            时间离散: 隐式欧拉
            求解器: 五对角线追赶法
            x_env: 一类边界值
            time_step: 时间步长
            decay_constant: 一阶化学反应系数
            boundary_transport_coef: 二类边界值
            x_k: 初始条件
            D: 扩散系数
            N: 总网格数
            r: 数值层半径
            volume: 网格体积
            area: 数值层面积
            gamma: 有限体积法中间参数
            Q: 数值层单位体积产生率
        '''
        N = volume.size
        h = np.array(r[1:] - r[0:-1])
        h = np.mean(h)
        D_right = np.append((D[0:N-1] + D[1:N])/2,0)
        D_left = np.append(0,(D[0:N-1]+D[1:N])/( 2))

        a = 4*D[2:N-2]/(r[2:N-2]+r[3:N-1])/12/h
        c = D[2:N-2]/12/h**2
        d = (8*D[3:N-1]-8*D[1:N-3]-D[4:N]+D[0:N-4])/(12*h)**2

        M = a-c+d
        G = -8*a + 16*c -8*d
        P = -30*c - decay_constant
        F = 8*a + 16*c + 8*d
        K = -a -c -d

        e1 = -2*D[1]/(r[2]+r[1])/h + D_left[1]/h**2
        f1 = -D_right[1]/h**2 - D_left[1]/h**2 - decay_constant
        g1 = 2*D[1]/(r[2]+r[1])/h + D_right[1]/h**2

        e_last = -2*D[N-1]/(r[N]+r[N-1])/h + D_left[N-1]/h**2
        f_last = -D[N-1]/h**2 - D_left[N-1]/h**2 - decay_constant
        g_last = 2*D[N-1]/(r[N-1]+r[N])/h + D[N-1]/h**2

        e_sub_last = -2*D[N-2]/(r[N-2]+r[N-1])/h + D_left[N-2]/h**2
        f_sub_last = -D_right[N-2]/h**2 - D_left[N-2]/h**2 - decay_constant
        g_sub_last = 2*D[N-2]/(r[N-2]+r[N-1]) + D_right[N-2]/h**2
        
        add = D[N-1]/h + boundary_transport_coef/2
        sub = D[N-1]/h - boundary_transport_coef/2

        central1 = 4*D[0]*h/r[1]**2/r[2] - 4*D[0]/r[1]**2 - D_right[0]/h**2 - decay_constant
        central2 = 4*D[0]*h/r[1]**2/r[2] + D_right[0]/h**2

        # 中心节点 中央差分
        diagonal3 = 1/time_step - central1
        diagonal4 = -central2
        diagonal5 = 0.0
        vector = (1/time_step)*x_k[0] + Q[0]

        # 次中心节点 中央差分
        diagonal2 = -e1
        diagonal3 = np.append(diagonal3,1/time_step-f1)
        diagonal4 = np.append(diagonal4,-g1)
        diagonal5 = np.append(diagonal5,0.0)
        vector = np.append(vector,x_k[1]/time_step + Q[1])

        # 中间节点 五点四阶
        diagonal1 = -M
        diagonal2 = np.append(diagonal2,-G)
        diagonal3 = np.append(diagonal3,1/time_step-P)
        diagonal4 = np.append(diagonal4,-F)
        diagonal5 = np.append(diagonal5,-K)
        vector = np.append(vector,x_k[2:N-2]/time_step+Q[2:N-2])

        # 倒数第二个节点 中央差分
        diagonal1 = np.append(diagonal1,0.0)
        diagonal2 = np.append(diagonal2,-e_sub_last)
        diagonal3 = np.append(diagonal3,1/time_step-f_sub_last)
        diagonal4 = np.append(diagonal4,-g_sub_last)
        vector = np.append(vector,x_k[N-2]/time_step + Q[N-2])

        # 最外侧节点 中央差分+Robin边界条件(虚拟节点)
        diagonal1 = np.append(diagonal1,0.0)
        diagonal2 = np.append(diagonal2,-e_last)
        diagonal3 = np.append(diagonal3,1/time_step-(f_last+g_last*sub/add))
        vector = np.append(vector,x_k[N-1]/time_step +g_last*boundary_transport_coef*x_env/add + Q[N-1])

        x_next = cls._penta_thomas(diagonal1,diagonal2,diagonal3,diagonal4,diagonal5,vector)
        x_image = sub/add*x_next[-1] + boundary_transport_coef*x_env/add
        x_nod = np.append(x_next[0] ,(x_next[0:-1]+x_next[1:])/(2))
        x_nod = np.append(x_nod,(x_image+x_next[-1])/2)
        return (x_next,x_nod)
    
    @classmethod
    def _fdmtaylor_trapezium_thomas(cls,x_env,time_step,decay_constant,boundary_transport_coef,x_k,D,r,volume,Q,w,area,gamma):
        '''
            空间离散: 高阶有限差分(五点四阶格式)
            时间离散: 隐式欧拉
            求解器: 五对角线追赶法
            x_env: 一类边界值
            time_step: 时间步长
            decay_constant: 一阶化学反应系数
            boundary_transport_coef: 二类边界值
            x_k: 初始条件
            D: 扩散系数
            N: 总网格数
            r: 数值层半径
            volume: 网格体积
            area: 数值层面积
            gamma: 有限体积法中间参数
            Q: 数值层单位体积产生率
            w: C-N格式前后时间层的分配系数
        '''
        N = volume.size
        h = np.array(r[1:] - r[0:-1])
        h = np.mean(h)
        D_right = np.append((D[0:N-1] + D[1:N])/2,0)
        D_left = np.append(0,(D[0:N-1]+D[1:N])/( 2))
        
        a = 4*D[2:N-2]/(r[2:N-2]+r[3:N-1])/12/h
        c = D[2:N-2]/12/h**2
        d = (8*D[3:N-1]-8*D[1:N-3]-D[4:N]+D[0:N-4])/(12*h)**2

        M = a-c+d
        G = -8*a + 16*c -8*d
        P = -30*c - decay_constant
        F = 8*a + 16*c + 8*d
        K = -a -c -d

        e1 = -2*D[1]/(r[2]+r[1])/h + D_left[1]/h**2
        f1 = -D_right[1]/h**2 - D_left[1]/h**2 - decay_constant
        g1 = 2*D[1]/(r[2]+r[1])/h + D_right[1]/h**2

        e_last = -2*D[N-1]/(r[N]+r[N-1])/h + D_left[N-1]/h**2
        f_last = -D[N-1]/h**2 - D_left[N-1]/h**2 - decay_constant
        g_last = 2*D[N-1]/(r[N-1]+r[N])/h + D[N-1]/h**2

        e_sub_last = -2*D[N-2]/(r[N-2]+r[N-1])/h + D_left[N-2]/h**2
        f_sub_last = -D_right[N-2]/h**2 - D_left[N-2]/h**2 - decay_constant
        g_sub_last = 2*D[N-2]/(r[N-2]+r[N-1]) + D_right[N-2]/h**2
        
        add = D[N-1]/h + boundary_transport_coef/2
        sub = D[N-1]/h - boundary_transport_coef/2

        central1 = 4*D[0]*h/r[1]**2/r[2] - 4*D[0]/r[1]**2 - D_right[0]/h**2 - decay_constant
        central2 = 4*D[0]*h/r[1]**2/r[2] + D_right[0]/h**2

        # 中心节点 中央差分+对称边界条件
        diagonal3 = 1/time_step - w*central1
        diagonal4 = -w*central2
        diagonal5 = 0.0
        vector = (1/time_step + (1-w)*central1)*x_k[0] + (1-w)*central2*x_k[1] + Q[0]

        # 次中心节点 中央差分
        diagonal2 = -w*e1
        diagonal3 = np.append(diagonal3,1/time_step - w*f1)
        diagonal4 = np.append(diagonal4,-w*g1)
        diagonal5 = np.append(diagonal5,0.0)
        vector = np.append(vector, (1/time_step+(1-w)*f1)*x_k[1] + (1-w)*e1*x_k[0] + (1-w)*g1*x_k[2] + Q[1] )

        # 中间节点 五点四阶
        diagonal1 = -w*M
        diagonal2 = np.append(diagonal2,-w*G)
        diagonal3 = np.append(diagonal3,1/time_step - w*P)
        diagonal4 = np.append(diagonal4,-w*F)
        diagonal5 = np.append(diagonal5,-w*K)
        vector = np.append(vector,(1-w)*M*x_k[0:N-4] + (1-w)*G*x_k[1:N-3] + (1/time_step + (1-w)*P)*x_k[2:N-2] + (1-w)*F*x_k[3:N-1] + (1-w)*K*x_k[4:N] + Q[2:N-2])

        # 倒数第二个节点 中央差分
        diagonal1 = np.append(diagonal1,0.0)
        diagonal2 = np.append(diagonal2, -w*e_sub_last)
        diagonal3 = np.append(diagonal3,1/time_step - w*f_sub_last)
        diagonal4 = np.append(diagonal4, -w*g_sub_last)
        vector = np.append(vector, (1-w)*e_sub_last*x_k[N-3] + (1/time_step +(1-w)*f_sub_last)*x_k[N-2] + (1-w)*g_sub_last*x_k[N-1] +Q[N-2] )

        # 最外侧节点 中央差分+Robin边界条件(虚拟节点)
        diagonal1 = np.append(diagonal1,0.0)
        diagonal2 = np.append(diagonal2, -w*e_last)
        diagonal3 = np.append(diagonal3, 1/time_step - w*(f_last+g_last*sub/add))
        vector = np.append(vector, (1-w)*e_last*x_k[N-2] + (1/time_step + (1-w)*(f_last+g_last*sub/add)*x_k[N-1] + g_last*boundary_transport_coef/add*x_env + Q[N-1]) )
        
        x_next = cls._penta_thomas(diagonal1,diagonal2,diagonal3,diagonal4,diagonal5,vector)   
        x_image = sub/add*x_next[-1] + boundary_transport_coef*x_env/add
        x_nod = np.append(x_next[0] ,(x_next[0:-1]+x_next[1:])/(2))
        x_nod = np.append(x_nod,(x_image+x_next[-1])/2)
        return (x_next,x_nod)

    @classmethod
    def calculate_numerical_solution(cls,solver,x_env,time_step,decay_constant,boundary_transport_coef,x_k_fdm,x_k_fvm,D,r,volume,Q,w,area,gamma):
        '''
            调用数值离散计算
        '''
        solver_dict = {
            'fvm_euler_thomas':cls._fvm_euler_thomas,
            'fvm_trapezium_thomas':cls._fvm_trapezium_thomas,
            'fdmleapfrog_euler_thomas':cls._fdmleapfrog_euler_thomas,
            'fdmleapfrog_trapezium_thomas':cls._fdmleapfrog_trapezium_thomas,
            'fdmtaylor_euler_thomas':cls._fdmtaylor_euler_thomas,
            'fdmtaylor_trapezium_thomas':cls._fdmtaylor_trapezium_thomas
        }
        x_k = x_k_fdm if 'fdm' in solver else x_k_fvm
        x_next_fdm,x_next_fvm = solver_dict[solver](x_env,time_step,decay_constant,boundary_transport_coef,x_k,D,r,volume,Q,w,area,gamma)
        
        return x_next_fdm,x_next_fvm
            

        
    

