
import numpy as np
from skimage.color import rgb2gray
from skimage import io, measure
from scipy import signal
from utility import *

metrics_name = ['qmi', 'qte', 'qncie','qg', 'qm', 'qsf', 'qp','qs', 'qc', 'qy','qcv', 'qcb',
                'viff','mef_ssim','ssim_a','fmi_edge', 'fmi_dct', 'fmi_w','nabf','scd','sd','sf', 'cc']

class Evaluator(MatlabEngine):
    """
    Evaluation Class
    The mertics 'Qmi', 'Qte', 'Qncie', 'Qg', 'Qm', 'Qsf', 'Qp', 'Qs', 'Qc', 'Qy', 'Qcv', 'Qcb'
    are copied from https://github.com/zhengliu6699/imageFusionMetrics
    and
    Liu Z, Blasch E, Xue Z, et al. Objective assessment of multiresolution image fusion algorithms
    for context enhancement in night vision: a comparative study[J]. IEEE transactions on pattern
    analysis and machine intelligence, 2011, 34(1): 94-109.
    The mertic 'VIFF' is copied from http://hansy.weebly.com/image-fusionmetric.html
    The mertic 'MEF_SSIM' is copied from https://github.com/hli1221/imagefusion_deeplearning
    and
    K. Ma, K. Zeng, and Z. Wang, “Perceptual quality assessment for multiexposure image fusion,”
    IEEE Trans. Image Process., vol. 24, no. 11,pp. 3345–3356, Nov. 2015.
    The mertic 'SSIM_A' is copied from
    DenseFuse: H. Li, X. J. Wu, “DenseFuse: A Fusion Approach to Infrared and Visible Images,” IEEE Trans.
    Image Process., vol. 28, no. 5, pp. 2614–2623, May. 2019.
    The metrics 'FMI_EDGE', 'FMI_DCT', 'FMI_W' are copied from https://github.com/hli1221/imagefusion_deeplearning
    and
    M. Haghighat, M.A. Razian, "Fast-FMI: non-reference image fusion metric,"
    8th International Conference on Application of Information and Communication Technologies (AICT), pp. 1-3, 2014.
    The metric 'Nabf' is copied from https://github.com/hli1221/imagefusion_deeplearning
    and
    objective_fusion_perform_fn: Computes the Objective Fusion Performance Parameters proposed by Petrovic
    and modified Fusion Artifacts (NABF) measure proposed by B. K. Shreyamsha Kumar
    The metric 'SCD' is copied from https://github.com/hli1221/imagefusion_deeplearning
    and
    V. Aslantas and E. Bendes, "A new image quality metric for image fusion: The sum of the correlations of differences,"
    AEU - International Journal of Electronics and Communications, vol. 69/12, pp. 1890-1896, 2015.
    The metrics 'SD', 'SF' , 'CC' are used by:
    Ma J, Yu W, Liang P, et al. FusionGAN: A generative adversarial network for infrared and visible image fusion[J].
    Information Fusion, 2019, 48: 11-26.
    The default parameters given in the related publications are adopted for these quality index
    All the metrics will assign a larger value to the better fusion result.
    Note:
        We find Qc and Qy may sometime return nan value, so we add 1e-10 on denominator in each
        division.
    """
    def __init__(self):
        super(Evaluator, self).__init__()

    def __del__(self):
        super(Evaluator, self).__del__()

    @staticmethod
    def np_to_mat(img):
        """
        Transfer numpy to matlab style
        :param img: image, np.array
        :return: matlab style
        """
        img_mat = matlab.double(img.tolist())
        return img_mat

    def get_evaluation(self, img1,img2,fused, metrics):
        """
        get evaluation by four metrics: qmi, qg, qy, qcb, the bigger, the better
        :param img1: last image, np.array
        :param img2: next image, np.array
        :param fused: fusion result, np.array
        :param metric_ids: which metric want to be calculated, list int range in [0,11]
        :return: q_values, list float
        """
        # Transfer RGB to Gray, it could be better to convert RGB to Gray before analyzing
        if img1.ndim == 3:     # note bgr in cv2 or rgb in skimage and matlab
            # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            # fused = cv2.cvtColor(fused, cv2.COLOR_BGR2GRAY)
            img1 = rgb2gray(img1)
            img2 = rgb2gray(img2)
            fused = rgb2gray(fused)

        # Transfer numpy to mat
        img1_mat = self.np_to_mat(img1)
        img2_mat = self.np_to_mat(img2)
        fused_mat = self.np_to_mat(fused)
        kwargs = {'img1':img1,
                  'img2':img2,
                  'fused':fused,
                  'img1_mat':img1_mat,
                  'img2_mat':img2_mat,
                  'fused_mat':fused_mat}
        # evaluation
        q_values = {}
        for metric in metrics:
            print(metric)
            q_values[metric] = round(self.__getattribute__('evaluate_by_'+metric)(img1=img1,
                                                                                          img2=img2,
                                                                                          fused=fused,
                                                                                          img1_mat=img1_mat,
                                                                                          img2_mat=img2_mat,
                                                                                          fused_mat=fused_mat),5)
        return q_values

    # ******************************** Information Theory-Based Metric ******************************************
    def evaluate_by_qmi(self, **kwargs):
        """
        Normalized Mutual Information (Qmi)
        M. Hossny, S. Nahavandi, and D. Vreighton, “Comments on‘Information Measure for Performance of
        Image Fusion’,” Electronics Letters, vol. 44, no. 18, pp. 1066-1067, Aug. 2008.
        As the paper described, the value is not accurate because it tends to become larger when the pixel
        values of the fused image are closer to one of the source images
        Liu Y, Chen X, Peng H, et al. Multi-focus image fusion with a deep convolutional neural network[J].
        Information Fusion, 2017, 36: 191-207.
        :param img1: last image, matlab mode
        :param img2: next image, matlab mode
        :param fused: fusion result, matlab mode
        :return: qmi,float
        """
        value = self.matlab_engine.metricMI(kwargs['img1_mat'], kwargs['img2_mat'], kwargs['fused_mat'], 1)  # sw = 1 revised MI
        return value

    def evaluate_by_qte(self, **kwargs):
        """
        Tsallis Entropy (Qte)
        N. Cvejic, C.N. Canagarajah, and D.R. Bull, “Image Fusion Metric Based on Mutual Information and
        Tsallis Entropy,” ElectronicsLetters, vol. 42, no. 11, pp. 626-627, May 2006.
        :param img1: last image, matlab mode
        :param img2: next image, matlab mode
        :param fused: fusion result, matlab mode
        :return: qte,float
        """
        value = self.matlab_engine.metricMI(kwargs['img1_mat'], kwargs['img2_mat'], kwargs['fused_mat'], 3)  # sw = 2 Tsallis Entropy
        return value

    def evaluate_by_qncie(self, **kwargs):
        """
        nonlinear correlation information entropy (Qncie)
        Q. Wang, Y. Shen, and J. Jin, “Performance Evaluation of Image Fusion Techniques,”
        Image Fusion: Algorithms and Applications, ch. 19, T. Stathaki, ed., pp. 469-492. Elsevier, 2008
        :param img1: last image, matlab mode
        :param img2: next image, matlab mode
        :param fused: fusion result, matlab mode
        :return: qncie,float
        """
        value = self.matlab_engine.metricWang(kwargs['img1_mat'], kwargs['img2_mat'], kwargs['fused_mat'])
        return value

    # ******************************** Image Feature-Based Metric ***********************************************
    def evaluate_by_qg(self, **kwargs):
        """
        Gradient-Based Fusion Performance(Qg)- evaluate the amount of edge information
        C.S. Xydeas and V. Petrovic, “Objective Image Fusion Performance Measure,”
        Electronics Letters, vol. 36, no. 4, pp. 308-309, 2000.
        :param img1: last image, matlab mode
        :param img2: next image, matlab mode
        :param fused: fusion result, matlab mode
        :return: qg,float
        """
        value = self.matlab_engine.metricXydeas(kwargs['img1_mat'], kwargs['img2_mat'], kwargs['fused_mat'])
        return value

    def evaluate_by_qm(self, **kwargs):
        """
        Multiscale scheme-Based Fusion Performance(Qm)
        P. Wang and B. Liu, “A Novel Image Fusion Metric Based on Multi-Scale Analysis,”
        Proc. IEEE Int’l Conf. Signal Processing, pp. 965-968, 2008.
        :param img1: last image, matlab mode
        :param img2: next image, matlab mode
        :param fused: fusion result, matlab mode
        :return: qm,float
        """
        value = self.matlab_engine.metricPWW(kwargs['img1_mat'], kwargs['img2_mat'], kwargs['fused_mat'])
        return value

    def evaluate_by_qsf(self, **kwargs):
        """
        Spatial Frequency-Based Fusion Performance Zheng's Metric(Qsf)
        Y. Zheng, E.A. Essock, B.C. Hansen, and A.M. Haun, “A New Metric Based on Extended Spatial
        Frequency and Its Application to DWT Based Fusion Algorithms,” Information Fusion, vol. 8, no.
        2, pp. 177-192, Apr. 2007.
        :param img1: last image, matlab mode
        :param img2: next image, matlab mode
        :param fused: fusion result, matlab mode
        :return: qsf,float
        """
        value = self.matlab_engine.metricZheng(kwargs['img1_mat'], kwargs['img2_mat'], kwargs['fused_mat'])
        # value = 1 - abs(value)
        return value

    def evaluate_by_qp(self, **kwargs):
        """
        Phase Congruency-Based Fusion Performance Zhao's Metric(Qp)
        J. Zhao, R. Laganiere, and Z. Liu, “Performance Assessment of Combinative Pixel-Level Image Fusion
        Based on an Absolute Feature Measurement,” Int’l J. Innovative Computing, Information and Control,
        vol. 3, no. 6(A), pp. 1433-1447, Dec. 2007
        :param img1: last image, matlab mode
        :param img2: next image, matlab mode
        :param fused: fusion result, matlab mode
        :return: qp,float
        """
        value = self.matlab_engine.metricZhao(kwargs['img1_mat'], kwargs['img2_mat'], kwargs['fused_mat'])
        return value

    # ******************************** Image Structural Similarity-Based Metrics ********************************
    def evaluate_by_qs(self, **kwargs):
        """
        Wang’s UIQI method-Based Fusion Performance Piella's Metric(Qs), we choose the sw = 1
        G. Piella and H. Heijmans, “A New Quality Metric for Image Fusion,” Proc.
        Int’l Conf. Image Processing, 2003.
        :param img1: last image, matlab mode
        :param img2: next image, matlab mode
        :param fused: fusion result, matlab mode
        :return: qs,float
        """
        value = self.matlab_engine.metricPeilla(kwargs['img1_mat'], kwargs['img2_mat'], kwargs['fused_mat'], 1)
        return value

    def evaluate_by_qc(self, **kwargs):
        """
        SSIM-Based Fusion Performance Cvejie's Metric(Qc), we choose the sw = 2
        We find metric Cvejic may sometime return nan value,
        so we add 1e-10 on denominator in each division.
        N. Cvejic, A. Loza, D. Bul, and N. Canagarajah, “A Similarity Metric for Assessment of Image Fusion
        Algorithms,” Int’l J. Signal Processing, vol. 2, no. 3, pp. 178-182, 2005.
        :param img1: last image, matlab mode
        :param img2: next image, matlab mode
        :param fused: fusion result, matlab mode
        :return: qc,float
        """
        value = self.matlab_engine.metricCvejic(kwargs['img1_mat'], kwargs['img2_mat'], kwargs['fused_mat'], 2)
        return value

    def evaluate_by_qy(self, **kwargs):
        """
        SSIM-Based Fusion Performance Yang's Metric(Qy)
        We find metricYang may sometime return nan value,
        so we add 1e-10 on denominator in each division.
         C. Yang, J. Zhang, X. Wang, and X. Liu, “A Novel Similarity Based Quality Metric for Image Fusion,”
         Information Fusion, vol. 9,pp. 156-160, 2008.
        :param img1: last image, matlab mode
        :param img2: next image, matlab mode
        :param fused: fusion result, matlab mode
        :return: qy,float
        """
        value = self.matlab_engine.metricYang(kwargs['img1_mat'], kwargs['img2_mat'], kwargs['fused_mat'])
        return value

    # ******************************** Human Perception Inspired Fusion Metrics ********************************
    def evaluate_by_qcv(self, **kwargs):
        """
        Human Perception inspired fusion metric - Chen - Varshney
        H. Chen and P.K. Varshney, “A Human Perception Inspired Quality Metric for Image Fusion Based on
        Regional Information,”Information Fusion, vol. 8, pp. 193-207, 2007
        :param img1: last image, matlab mode
        :param img2: next image, matlab mode
        :param fused: fusion result, matlab mode
        :return: Qcv,float
        """
        value = self.matlab_engine.metricChen(kwargs['img1_mat'], kwargs['img2_mat'], kwargs['fused_mat'])
        return value

    def evaluate_by_qcb(self, **kwargs):
        """
        Human Perception inspired fusion metric - chen blum Metric
        Using DoG as contrast preservation filter
        Y. Chen and R.S. Blum, “A New Automated Quality Assessment
        Algorithm for Image Fusion,” Image and Vision Computing, vol. 27,pp. 1421-1432, 2009.
        :param img1: last image, matlab mode
        :param img2: next image, matlab mode
        :param fused: fusion result, matlab mode
        :return: Qcb,float
        """
        value = self.matlab_engine.metricChenBlum(kwargs['img1_mat'], kwargs['img2_mat'], kwargs['fused_mat'])
        return value

    # ************************************************** VIFF **************************************************
    def evaluate_by_viff(self, **kwargs):
        """
        A multi-resolution image fusion metric using visual information fidelity (VIF)
        Y. Han, Y. Cai, Y. Cao, and X. Xu, “A new image fusion performance metric based on
        visual information fidelity,” Information fusion, vol. 14,no. 2, pp. 127–135, 2013.
        :param img1: last image, matlab mode
        :param img2: next image, matlab mode
        :param fused: fusion result, matlab mode
        :return: VIFF,float
        """
        value = self.matlab_engine.VIFF_Public(kwargs['img1_mat'], kwargs['img2_mat'], kwargs['fused_mat'])
        return value

    # ************************************************** mef-SSIM ************************************************
    def evaluate_by_mef_ssim(self, **kwargs):
        """
        We copy code from https://github.com/hli1221/imagefusion_deeplearning
        K. Ma, K. Zeng, and Z. Wang, “Perceptual quality assessment for multiexposure image fusion,”
        IEEE Trans. Image Process., vol. 24, no. 11,pp. 3345–3356, Nov. 2015.
        :param img1: last image, matlab mode
        :param img2: next image, matlab mode
        :param fused: fusion result, matlab mode
        :return: mef_ssim, float
        """
        value = self.matlab_engine.mef_ssim(kwargs['img1_mat'], kwargs['img2_mat'], kwargs['fused_mat'])
        return value

    # *************************************************** SSIM_A ************************************************
    def evaluate_by_ssim_a(self, **kwargs):
        """
        DenseFuse: H. Li, X. J. Wu, “DenseFuse: A Fusion Approach to Infrared and Visible Images,” IEEE Trans.
        Image Process., vol. 28, no. 5, pp. 2614–2623, May. 2019.
        :param img1: last image, ndarray
        :param img2: next image, ndarray
        :param fused: fusion result, ndarray
        :return: ssim_a,float
        """
        im1_f_ssim = measure.compare_ssim(kwargs['img1'],kwargs['fused'])
        im2_f_ssim = measure.compare_ssim(kwargs['img2'], kwargs['fused'])
        return (im1_f_ssim + im2_f_ssim) / 2

    # **************************************************** FMI ************************************************
    def evaluate_by_fmi_edge(self, **kwargs):
        """
        We copy code from https://github.com/hli1221/imagefusion_deeplearning
        M. Haghighat, M.A. Razian, "Fast-FMI: non-reference image fusion metric,"
        8th International Conference on Application of Information and Communication Technologies (AICT), pp. 1-3, 2014.
        'edge' mode
        :param img1: last image, matlab mode
        :param img2: next image, matlab mode
        :param fused: fusion result, matlab mode
        :return: fmi_edge, float
        """
        value = self.matlab_engine.analysis_fmi(kwargs['img1_mat'], kwargs['img2_mat'], kwargs['fused_mat'])
        return value

    def evaluate_by_fmi_dct(self, **kwargs):
        """
        We copy code from https://github.com/hli1221/imagefusion_deeplearning
        M. Haghighat, M.A. Razian, "Fast-FMI: non-reference image fusion metric,"
        8th International Conference on Application of Information and Communication Technologies (AICT), pp. 1-3, 2014.
        'dct' mode
        :param img1: last image, matlab mode
        :param img2: next image, matlab mode
        :param fused: fusion result, matlab mode
        :return: fmi_dct, float
        """
        value = self.matlab_engine.analysis_fmi(kwargs['img1_mat'], kwargs['img2_mat'], kwargs['fused_mat'], 'dct')
        return value

    def evaluate_by_fmi_w(self, **kwargs):
        """
        We copy code from https://github.com/hli1221/imagefusion_deeplearning
        M. Haghighat, M.A. Razian, "Fast-FMI: non-reference image fusion metric,"
        8th International Conference on Application of Information and Communication Technologies (AICT), pp. 1-3, 2014.
        'wavelet' mode
        :param img1: last image, matlab mode
        :param img2: next image, matlab mode
        :param fused: fusion result, matlab mode
        :return: fmi_w, float
        """
        value = self.matlab_engine.analysis_fmi(kwargs['img1_mat'], kwargs['img2_mat'], kwargs['fused_mat'], 'wavelet')
        return value

    # **************************************************** Nabf ************************************************
    def evaluate_by_nabf(self, **kwargs):
        """
        We copy code from https://github.com/hli1221/imagefusion_deeplearning
        objective_fusion_perform_fn: Computes the Objective Fusion Performance Parameters proposed by Petrovic
        and modified Fusion Artifacts (NABF) measure proposed by B. K. Shreyamsha Kumar
        :param img1: last image, matlab mode
        :param img2: next image, matlab mode
        :param fused: fusion result, matlab mode
        :return: nabf, float
        """
        value = self.matlab_engine.analysis_nabf(kwargs['img1_mat'], kwargs['img2_mat'], kwargs['fused_mat'])
        return value

    # **************************************************** SCD ************************************************
    def evaluate_by_scd(self,**kwargs):
        """
        We copy code from https://github.com/hli1221/imagefusion_deeplearning
        V. Aslantas and E. Bendes, "A new image quality metric for image fusion:
        The sum of the correlations of differences,"  AEU - International Journal of Electronics and Communications, vol. 69/12, pp. 1890-1896, 2015.
        However We found that value would be nan, please be careful about that
        :param img1: last image, matlab mode
        :param img2: next image, matlab mode
        :param fused: fusion result, matlab mode
        :return: scd, float
        """
        value = self.matlab_engine.analysis_SCD(kwargs['img1_mat'], kwargs['img2_mat'], kwargs['fused_mat'])
        return value

    # **************************************************** SD ************************************************
    def evaluate_by_sd(self, **kwargs):
        """
        standard deviation
        Ma J, Yu W, Liang P, et al. FusionGAN: A generative adversarial network for infrared and visible image fusion[J].
        Information Fusion, 2019, 48: 11-26.
        :param fused: fusion result, ndarray
        :return: sd,float
        """
        fused_mean = np.mean(kwargs['fused'])
        value = np.sqrt(np.sum(np.power(kwargs['fused']- fused_mean, 2)))
        return value

    # **************************************************** SF ************************************************
    def evaluate_by_sf(self, **kwargs):
        """
        spatial frequency
        Ma J, Yu W, Liang P, et al. FusionGAN: A generative adversarial network for infrared and visible image fusion[J].
        Information Fusion, 2019, 48: 11-26.
        :param fused: fusion result, ndarray
        :return: sf,float
        """
        # r_shift_kernel = np.expand_dims(np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]]), axis=2)  # right shift kernel
        # b_shift_kernel = np.expand_dims(np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]), axis=2)  # bottom shift kernel
        r_shift_kernel = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]])  # right shift kernel
        b_shift_kernel = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])  # bottom shift kernel

        fused_r_shift = signal.correlate(kwargs['fused'], r_shift_kernel, mode='same')
        fused_b_shift = signal.correlate(kwargs['fused'], b_shift_kernel, mode='same')
        temp_sum = np.sum(np.power(kwargs['fused'] - fused_r_shift, 2).astype(np.int64) + np.power(kwargs['fused'] - fused_b_shift, 2).astype(np.int64))
        sf = np.sqrt(temp_sum)
        return sf

    # **************************************************** CC ************************************************
    def evaluate_by_cc(self, **kwargs):
        """
        linear correlation of the fused image and source images
        Ma J, Yu W, Liang P, et al. FusionGAN: A generative adversarial network for infrared and visible image fusion[J].
        Information Fusion, 2019, 48: 11-26.
        :param img1: last image, ndarray
        :param img2: next image, ndarray
        :param fused: fusion result, ndarray
        :return: cc,float
        """
        r1f = self._calculate_rxf(kwargs['img1'], kwargs['fused'])
        r2f = self._calculate_rxf(kwargs['img2'], kwargs['fused'])
        return (r1f + r2f)/2

    @staticmethod
    def _calculate_rxf(img, fused):
        """
        We implement fomula (11)
        :param img: ndarray
        :param fused: fusion result, ndarray
        :return: rxf
        """
        img_mean = np.mean(img)
        fused_mean = np.mean(fused)
        corr = np.sum((img - img_mean) * (fused - fused_mean))
        temp_f = np.sum(np.power(fused - fused_mean, 2))
        value = corr / np.sqrt(np.sum(np.power(img - img_mean, 2)* temp_f))
        return value
