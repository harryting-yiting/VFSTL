from lib_stl_core import *

'''
    J = states[:,:, 0]
    W = states[:,:, 1]
    R = states[:,:, 2]
    Y = states[:,:, 3]
'''

# we considering V >= 0.8 as complete the task
threshold = 0.9
avoid_treshold = 0.2
def reach_avoid_stl_formula(nt):
    '''
       reach red while avoiding black 
    '''
    Reach = Eventually(0, nt, AP(lambda x: x[..., 2] - threshold, comment="REACH RED"))
    Avoid = Always(0, nt, AP(lambda x: threshold - x[..., 0], comment="AVOID BLACK"))               # TODO: adjust threshold for avoid
    stl = ListAnd([Reach, Avoid])

    # check the STL content
    stl.update_format("word")
    print(stl)

    return stl

def sequential_reach_stl_formula(nt):
    '''
        reach red, reach yellow, reach black
    '''
    ReachRed = Eventually(0, nt//3, AP(lambda x: x[..., 2] - threshold, comment="REACH RED"))
    ReachYellow = Eventually(nt//3+1, 2*nt//3, AP(lambda x: x[..., 3] - threshold, comment="REACH YELLOW"))
    ReachBlack = Eventually(2*nt//3+1, nt, AP(lambda x: x[..., 0] - threshold, comment="REACH BLACK"))
    stl = ListAnd([ReachRed, ReachYellow, ReachBlack])

    # check the STL content
    stl.update_format("word")
    print(stl)

    return stl

def reach_black_avoid_others_stl_formula(nt):
    ReachBlack = Eventually(0, nt, AP(lambda x: x[..., 0] - threshold, comment="REACH BLACK"))
    AvoidRed = Always(0, nt, AP(lambda x: avoid_treshold - x[..., 2], comment="AVOID Red"))
    AvoidYellow = Always(0, nt, AP(lambda x: avoid_treshold - x[..., 3], comment="AVOID Yellow"))
    AvoidWhite = Always(0, nt, AP(lambda x: avoid_treshold - x[..., 1], comment="AVOID White"))
    stl = ListAnd([ReachBlack, AvoidRed, AvoidYellow, AvoidWhite])

    # check the STL content
    stl.update_format("word")
    print(stl)

    return stl


def sequential_avoid_stl_formula(nt):
    '''
       reach red, reach yellow, while avoiding black
    '''
    ReachRed = Eventually(0, nt//2, AP(lambda x: x[..., 2] - threshold, comment="REACH RED"))
    ReachYellow = Eventually(nt//2+1, nt, AP(lambda x: x[..., 3] - threshold, comment="REACH YELLOW"))
    Avoid = Always(0, nt, AP(lambda x: threshold - x[..., 0], comment="AVOID BLACK"))               # TODO: adjust threshold for avoid
    stl = ListAnd([ReachRed, ReachYellow, Avoid])

    # check the STL content
    stl.update_format("word")
    print(stl)

    return stl
