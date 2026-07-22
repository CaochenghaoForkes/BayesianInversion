import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.sparse.linalg import inv
from TOOL.cosmos_general_function import Registry
from ACCIDENT.accident_xml_reader import AccidentXMLReader

class AccidentSolver:
    
    @classmethod
    def _mmpa_32(cls, x, A, t):

        alpha = np.array([
            3.41366034346810763441E-11,
            -1.64648752802981232517E-09,
            3.80465674795813706161E-08,
            -5.59799382133376361074E-07,
            5.88558416643299794299E-06,
            -4.69954781431498324372E-05,
            2.95331252981155031018E-04,
            -1.49342612561658940208E-03,
            6.16185418583114456893E-03,
            -2.08884457395397287047E-02,
            5.81666843649626684195E-02,
            -1.31988170767468710376E-01,
            2.39594549098354306472E-01,
            -3.34580286145275369457E-01,
            3.25793940419810518947E-01,
            -1.47667478685995070947E-01,
            -1.19600677614316509554E-01,
            2.40742697969870662426E-01,
            -7.60967222595908615166E-02,
            -1.58220954609283804027E-01,
            1.32429922343368914663E-01,
            7.69052976527847896194E-02,
            -1.14189797653561609782E-01,
            -3.06105871275143780464E-02,
            7.30162061991511174207E-02,
            9.95181782801371819122E-03,
            -3.55155454832496982106E-02,
            -2.4887047789165500977E-03,
            1.23929263457145990096E-02,
            4.21590406862371374909E-04,
            -2.74111882594745761616E-03,
            -3.57929539250313916141E-05,
            2.86523961626939505591E-04
        ], dtype=np.float64)

        c = 24.1

        # 稀疏矩阵形式
        A_sp = sp.csc_matrix(A, dtype=np.float64)
        I = sp.eye(A_sp.shape[0], format='csc', dtype=np.float64)
        At = t * A_sp

        part_2 = At + c * I          # (At + cI)
        part_3 = At - c * I          # (At - cI)

        # 预分解 (At - cI) 以便多次求解时更快
        # 如果只调用一次，可以直接用 spsolve，频繁调用的话用 splu/factorized 会更划算
        solver = sla.factorized(part_3)   # solver(v) 相当于解 part_3 z = v

        def apply_M(v):
            # z = (At - cI)^(-1) v
            z = solver(v)
            # y = (At + cI) z
            return part_2 @ z

        # Horner 法计算 p(M) x，p(z) = Σ alpha[i] z^i
        # y = alpha_n * x
        y = alpha[-1] * x.copy()

        # 从 alpha[31] 到 alpha[0]
        for coef in alpha[-2::-1]:
            y = apply_M(y) + coef * x

        return y
    
    @classmethod
    def _cram_14(cls,x,A,t):

        theta = np.array([
            -8.8977731864688888199e+0 + 1.6630982619902085304e+1j,
            -3.7032750494234480603e+0 + 1.3656371871483268171e+1j,
            -0.2087586382501301251e+0 + 1.0991260561901260913e+1j,
            +3.9933697105785685194e+0 + 6.0048316422350373178e+0j,
            +5.0893450605806245066e+0 + 3.5888240290270065102e+0j,
            +5.6231425727459771248e+0 + 1.1940690463439669766e+0j,
            +2.2697838292311127097e+0 + 8.4617379730402214019e+0j 
        ],dtype=np.complex128)

        alpha = np.array([
            -7.1542880635890672853e-5 + 1.4361043349541300111e-4j,
            +9.4390253107361688779e-3 - 1.7184791958483017511e-2j,
            -3.7636003878226968717e-1 + 3.3518347029450104214e-1j,
            -2.3498232091082701191e+1 - 5.8083591297142074004e+0j,
            +4.6933274488831293047e+1 + 4.5643649768827760791e+1j,
            -2.7875161940145646468e+1 - 1.0214733999056451434e+2j,
            +4.8071120988325088907e+0 - 1.3209793837428723881e+0j
        ],dtype=np.complex128)

        alpha0 = 1.8321743782540412751e-14
        
        # 确保 x 是实数向量
        x = np.asarray(x, dtype=np.float64)

        # 用复数稀疏矩阵
        A_sp = sp.csc_matrix(A, dtype=np.complex128)
        n_dim = A_sp.shape[0]
        I = sp.eye(n_dim, format='csc', dtype=np.complex128)
        At = t * A_sp

        # 预 factorize，每个 theta 对应一个求解器
        solvers = []
        for th in theta:
            M = At - th * I               # (At - theta_j I)
            solvers.append(sla.factorized(M))

        # 复数累加
        n = np.zeros_like(x, dtype=np.complex128)
        for a, solve in zip(alpha, solvers):
            n += a * solve(x)

        # 最终结果：2*Re(Σ α_j (At-θ_jI)^(-1) x) + α0 x
        return 2.0 * n.real + alpha0 * x
    
    @classmethod
    def _cram_16(cls, x, A, t):

        theta = np.array([
            -1.0843917078696988026e+1 + 1.9277446167181652284e+1j,
            -5.2649713434426468895e+0 + 1.6220221473167927305e+1j,
            +5.9481522689511774808e+0 + 3.5874573620183222829e+0j,
            +3.5091036084149180974e+0 + 8.4361989858843750826e+0j,
            +6.4161776990994341923e+0 + 1.1941223933701386874e+0j,
            +1.4193758971856659786e+0 + 1.0925363484496722585e+1j,
            +4.9931747377179963991e+0 + 5.9968817136039422260e+0j,
            -1.4139284624888862114e+0 + 1.3497725698892745389e+1j
        ], dtype=np.complex128)

        alpha = np.array([
            -5.0901521865224915650e-7 - 2.4220017652852287970e-5j,
            +2.1151742182466030907e-4 + 4.3892969647380673918e-3j,
            +1.1339775178483930527e+2 + 1.0194721704215856450e+2j,
            +1.5059585270023467528e+1 - 5.7514052776421819979e+0j,
            -6.4500878025539646595e+1 - 2.2459440762652096056e+2j,
            -1.4793007113557999718e+0 + 1.7686588323782937906e+0j,
            -6.2518392463207918892e+1 - 1.1190391094283228480e+1j,
            +4.1023136835410021273e-2 - 1.5743466173455468191e-1j
        ], dtype=np.complex128)

        alpha0 = 2.1248537104952237488e-16

        # 确保 x 是实数向量
        x = np.asarray(x, dtype=np.float64)

        # 用复数稀疏矩阵
        A_sp = sp.csc_matrix(A, dtype=np.complex128)
        n_dim = A_sp.shape[0]
        I = sp.eye(n_dim, format='csc', dtype=np.complex128)
        At = t * A_sp

        # 预 factorize，每个 theta 对应一个求解器
        solvers = []
        for th in theta:
            M = At - th * I               # (At - theta_j I)
            solvers.append(sla.factorized(M))

        # 复数累加
        n = np.zeros_like(x, dtype=np.complex128)
        for a, solve in zip(alpha, solvers):
            n += a * solve(x)

        # 最终结果：2*Re(Σ α_j (At-θ_jI)^(-1) x) + α0 x
        return 2.0 * n.real + alpha0 * x

    @classmethod
    def _cram_16_openmc(cls,x,A,t):

        # Coefficients for IPF Cram 16
        c16_alpha = np.array([
            +5.464930576870210e+3 - 3.797983575308356e+4j,
            +9.045112476907548e+1 - 1.115537522430261e+3j,
            +2.344818070467641e+2 - 4.228020157070496e+2j,
            +9.453304067358312e+1 - 2.951294291446048e+2j,
            +7.283792954673409e+2 - 1.205646080220011e+5j,
            +3.648229059594851e+1 - 1.155509621409682e+2j,
            +2.547321630156819e+1 - 2.639500283021502e+1j,
            +2.394538338734709e+1 - 5.650522971778156e+0j],
            dtype=np.complex128)

        c16_theta = np.array([
            +3.509103608414918 + 8.436198985884374j,
            +5.948152268951177 + 3.587457362018322j,
            -5.264971343442647 + 16.22022147316793j,
            +1.419375897185666 + 10.92536348449672j,
            +6.416177699099435 + 1.194122393370139j,
            +4.993174737717997 + 5.996881713603942j,
            -1.413928462488886 + 13.49772569889275j,
            -10.84391707869699 + 19.27744616718165j],
            dtype=np.complex128)

        c16_alpha0 = 2.124853710495224e-16

        alpha = c16_alpha
        theta = c16_theta
        alpha0 = c16_alpha0
        A = t * sp.csc_matrix(A, dtype=np.float64)
        y = x.copy()
        ident = sp.eye(A.shape[0], format='csc')
        for alpha, theta in zip(alpha, theta):
            y += 2*np.real(alpha*sla.spsolve(A - theta*ident, y))
        
        return y * alpha0

    @classmethod
    def _cram_48_openmc(cls,x,A,t):

        theta_r = np.array([
            -4.465731934165702e+1, -5.284616241568964e+0,
            -8.867715667624458e+0, +3.493013124279215e+0,
            +1.564102508858634e+1, +1.742097597385893e+1,
            -2.834466755180654e+1, +1.661569367939544e+1,
            +8.011836167974721e+0, -2.056267541998229e+0,
            +1.449208170441839e+1, +1.853807176907916e+1,
            +9.932562704505182e+0, -2.244223871767187e+1,
            +8.590014121680897e-1, -1.286192925744479e+1,
            +1.164596909542055e+1, +1.806076684783089e+1,
            +5.870672154659249e+0, -3.542938819659747e+1,
            +1.901323489060250e+1, +1.885508331552577e+1,
            -1.734689708174982e+1, +1.316284237125190e+1])

        theta_i = np.array([
            +6.233225190695437e+1, +4.057499381311059e+1,
            +4.325515754166724e+1, +3.281615453173585e+1,
            +1.558061616372237e+1, +1.076629305714420e+1,
            +5.492841024648724e+1, +1.316994930024688e+1,
            +2.780232111309410e+1, +3.794824788914354e+1,
            +1.799988210051809e+1, +5.974332563100539e+0,
            +2.532823409972962e+1, +5.179633600312162e+1,
            +3.536456194294350e+1, +4.600304902833652e+1,
            +2.287153304140217e+1, +8.368200580099821e+0,
            +3.029700159040121e+1, +5.834381701800013e+1,
            +1.194282058271408e+0, +3.583428564427879e+0,
            +4.883941101108207e+1, +2.042951874827759e+1])

        c48_theta = np.array(theta_r + theta_i * 1j, dtype=np.complex128)

        alpha_r = np.array([
            +6.387380733878774e+2, +1.909896179065730e+2,
            +4.236195226571914e+2, +4.645770595258726e+2,
            +7.765163276752433e+2, +1.907115136768522e+3,
            +2.909892685603256e+3, +1.944772206620450e+2,
            +1.382799786972332e+5, +5.628442079602433e+3,
            +2.151681283794220e+2, +1.324720240514420e+3,
            +1.617548476343347e+4, +1.112729040439685e+2,
            +1.074624783191125e+2, +8.835727765158191e+1,
            +9.354078136054179e+1, +9.418142823531573e+1,
            +1.040012390717851e+2, +6.861882624343235e+1,
            +8.766654491283722e+1, +1.056007619389650e+2,
            +7.738987569039419e+1, +1.041366366475571e+2])

        alpha_i = np.array([
            -6.743912502859256e+2, -3.973203432721332e+2,
            -2.041233768918671e+3, -1.652917287299683e+3,
            -1.783617639907328e+4, -5.887068595142284e+4,
            -9.953255345514560e+3, -1.427131226068449e+3,
            -3.256885197214938e+6, -2.924284515884309e+4,
            -1.121774011188224e+3, -6.370088443140973e+4,
            -1.008798413156542e+6, -8.837109731680418e+1,
            -1.457246116408180e+2, -6.388286188419360e+1,
            -2.195424319460237e+2, -6.719055740098035e+2,
            -1.693747595553868e+2, -1.177598523430493e+1,
            -4.596464999363902e+3, -1.738294585524067e+3,
            -4.311715386228984e+1, -2.777743732451969e+2])

        c48_alpha = np.array(alpha_r + alpha_i * 1j, dtype=np.complex128)

        c48_alpha0 = 2.258038182743983e-47

        alpha = c48_alpha
        theta = c48_theta
        alpha0 = c48_alpha0
        A = t * sp.csc_matrix(A, dtype=np.float64)
        y = x.copy()
        ident = sp.eye(A.shape[0], format='csc')
        for alpha, theta in zip(alpha, theta):
            y += 2*np.real(alpha*sla.spsolve(A - theta*ident, y))
        
        return y * alpha0

    @classmethod
    def _qram(cls,x,A,t,order):

        N = order
        theta = np.pi * (np.arange(1, N+1) - 0.5) / N
        z = N * (0.1309 - 0.1194*(theta**2) + (0.25j)*theta)
        dz_dtheta = N * (-0.1194 * 2 * theta + 0.25j)
        c = -(1j / N) * dz_dtheta
        n = A.shape[0]
        I = np.eye(n, dtype=complex)
        V = np.zeros_like(A, dtype=complex) 
        for k in range(N):
            V += c[k] * np.exp(z[k]) * np.linalg.inv((z[k]*I-A*t))

        result = V.real @ x

        return result

    def __init__(self,reader:AccidentXMLReader,register:Registry):
        # 模型注册
        register.register('matrix_solver','CRAM16',self._cram_16)
        register.register('matrix_solver','CRAM14',self._cram_14)
        register.register('matrix_solver','CRAMOpenMC16',self._cram_16_openmc)
        register.register('matrix_solver','CRAMOpenMC48',self._cram_48_openmc)
        register.register('matrix_solver','MMPA32',self._mmpa_32)
        register.register('matrix_solver','QRAM',self._qram)
        self.matrix_solver = reader.solver_dict['matrix_solver']
        if 'QRAM' in self.matrix_solver:
            self.order_qram = int(self.matrix_solver[4:])
    