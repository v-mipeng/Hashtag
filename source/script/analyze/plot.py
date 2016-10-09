import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D as line

def plot_hit_by_rank():
    ranks = np.arange(1,11)
    FD = [0.03327281, 0.053608715, 0.066681798, 0.073990014, 0.079119383, 0.084611893, 0.089605084, 0.099228325, 0.103268271, 0.108125284]
    FD2_005 = [0.088651838, 0.146890604, 0.180980481, 0.204720835, 0.221425329, 0.233817522, 0.244303223, 0.255469814,
     0.264457558, 0.276486609]
    FD2_1 = [0.001679528, 0.002178847, 0.003222878, 0.004085338, 0.004857013, 0.011938266, 0.013436223, 0.020426691, 0.021016795, 0.021334544]
    TopicWA = [0.050839764, 0.079300953, 0.097503404, 0.110031775, 0.118747163, 0.126509305, 0.132047208, 0.137131185, 0.141398094, 0.145937358]
    CNN_Attention = [0.103268271, 0.114117113, 0.11938266, 0.124602814, 0.128461189, 0.131093963, 0.133499773, 0.134906945, 0.136858829, 0.138992283]
    B_LSTM = [0.0920108942351,0.111211983659,0.123558783477,0.131911030413,0.138856105311,0.144484793463,0.149205628688,0.153200181571,0.157421697685,0.160054471176]
    A_LSTM=[0.17241488878800001, 0.20686790739900004, 0.225115751248, 0.23805265547, 0.24980935088499998, 0.25738992283299994, 0.263381752156, 0.268057194734, 0.27314117113,0.277453473]
    A_LSTM_Attention = [0.21041761234700002, 0.24264639128499998, 0.262800726282, 0.277008624603, 0.286450295052, 0.294757149342, 0.30106672719, 0.307557875624, 0.312414888788, 0.31654561961]

    #region Reference
    # markers = {0: 'tickleft', 1: 'tickright', 'v': 'triangle_down', 3: 'tickdown', 4: 'caretleft',
    #  5: 'caretright', 'd': 'thin_diamond', 'o': 'circle', 2: 'tickup', '3': 'tri_left',
    # ',': 'pixel', '.': 'point', '_': 'hline', ' ': 'nothing', 'H': 'hexagon2', 7: 'caretdown',
    # None: 'nothing', '': 'nothing', 'p': 'pentagon', 's': 'square', 'D': 'diamond', '|': 'vline',
    #  6: 'caretup', '8': 'octagon', 'None': 'nothing', '*': 'star', '+': 'plus', '<': 'triangle_left',
    #  'x': 'x', 'h': 'hexagon1', '1': 'tri_down', '>': 'triangle_right', '2': 'tri_up', '^': 'triangle_up',
    #  '4': 'tri_right'}
    #endregion
    models = (FD, FD2_005, FD2_1, TopicWA, CNN_Attention, B_LSTM, A_LSTM, A_LSTM_Attention)
    markers = ('x','s','^','*','o','2', '+','D')
    marker_sizes = (5,5,5,10,5,10, 10,5)
    marker_colors = ('k','w','w','w','w','w','w','w')
    labels = ('FD',r'FD2$(\alpha=0.005)$',r'FD2$(\alpha=1)$', 'TopicWA', 'CNN-Attention','B-LSTM', 'A-LSTM', 'A-LSTM-Shifting')

    for model, marker, marker_color, marker_size, label in zip(models, markers, marker_colors, marker_sizes, labels):
        line = plt.plot(ranks, model)[0]
        line.set_linestyle('-')
        line.set_color('k')
        line.set_linewidth(1)
        line.set_markeredgewidth(1)
        line.set_markersize(marker_size)
        line.set_marker(marker)
        line.set_markerfacecolor(marker_color)
        line.set_label(label)

    plt.axis([0,11,0,0.35])
    plt.legend(loc='upper left', fontsize='small')

    x = np.arange(0,11,0.1)
    y = np.arange(0.00,0.50,0.05)
    for v in y:
        dot_line = plt.plot(x,v*np.ones_like(x),'.')[0]
        dot_line.set_color('k')
        dot_line.set_markersize(0.5)
    plt.show()


def plot_kl_cr():
    akls = [0.41503739284850949,  0.42843982036397954,  0.44332363600927743,  0.45749269377261381,  0.46808751596188825,  0.47772175399653699,  0.47501136902130892,  0.47998968508534884,  0.48903097747245017,  0.49455839350191866,  0.49712779890346342,  0.50255181398206117,  0.5070277118560389,  0.50735274328028612,  0.51164491301946657,  0.51774856501286304,  0.52202587480505025,  0.522564219372738,  0.52577485598515694,  0.53001053237684181,  0.52917539549665527,  0.53172929049914408,  0.53597179202032563,  0.53762473096888341,  0.5387021706185996,  0.54210528180065187,  0.54496231975684095,  0.54711122354285047,  0.55121963584802547]
    acrs = [0.5358049126250007,  0.6102081653630826,  0.650288599307756,  0.6774425712744907,  0.6966282436425352,  0.7137485194266728,  0.7292190221793412,  0.7381391421404502,  0.7455124566689804,  0.7517145056447155,  0.757809661067203,  0.7627063705041269,  0.7675103564075858,  0.7728977702907978,  0.7772005761266934,  0.7806972953744,  0.7843211702075067,  0.7873012608064622,  0.7896081352194747,  0.7922368443325378,  0.7952835221640682,  0.7975932928341405,  0.799307028377616,  0.8014507440937723,  0.8034477267907665,  0.8051897508952227,  0.8069376117198664,  0.80878686056788,  0.8103934212315542]
    gaps = np.arange(1,len(akls)+1)

    #plot akl
    line = plt.plot(gaps, akls,'k-x')[0]
    line.set_linewidth(2)
    line.set_markersize(7)
    line.set_markeredgewidth(2)
    line.set_label(r'$AKL$')
    #plot acr
    line = plt.plot(gaps, acrs,'k-^')[0]
    line.set_linewidth(2)
    line.set_markersize(8)
    # line.set_markerfacecolor('w')
    line.set_label(r'$ACH$')

    x = np.arange(0,30,0.1)
    y = np.arange(0.5,0.8,0.1)
    for v in y:
        dot_line = plt.plot(x,v*np.ones_like(x),'.')[0]
        dot_line.set_color('k')
        dot_line.set_markersize(0.5)
    plt.legend(loc='upper left')
    plt.show()

if __name__=="__main__":
    plot_hit_by_rank()

