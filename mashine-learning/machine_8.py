'''
本案例用来测试 遗传算法: 使用遗传算法解决背包问题
'''

# # coding=utf-8
# import random
#
# # 背包问题
# # 物品重量价格
# X = {
#     1: [10, 15],
#     2: [15, 20],
#     3: [20, 35],
#     4: [25, 45],
#     5: [30, 55],
#     6: [35, 70]
# }
#
# # 终止界限
# FINISHED_LIMIT = 5
#
# # 重量界限
# WEIGHT_LIMIT = 80
#
# # 染色体长度
# CHROMOSOME_SIZE = 6
#
# # 遴选次数
# SELECT_NUMBER = 4
#
# max_last = 0
# diff_last = 10000
#
# # 收敛条件 判断退出
# def is_finished(fitnesses):
#     global max_last
#     global diff_last
#
#     max_current = 0
#     for v in fitnesses:
#         if v[1] > max_current:
#             max_current = v[1]
#
#     diff = max_current - max_last
#     if diff < FINISHED_LIMIT and diff_last < FINISHED_LIMIT:
#         return True
#     else:
#         diff_last = diff
#         max_last = max_current
#         return False
#
# # 初始化染色体状态
# def init():
#     chromosome_state1 = '100100'
#     chromosome_state2 = '101010'
#     chromosome_state3 = '010101'
#     chromosome_state4 = '101011'
#     chromosome_states = [
#         chromosome_state1,
#         chromosome_state2,
#         chromosome_state3,
#         chromosome_state4
#     ]
#     return chromosome_states
#
# # 计算适应度
# def fitness(chromosome_states):
#     fitnesses = []
#     for chromosome_state in chromosome_states:
#         value_sum = 0
#         weight_sum = 0
#         for i, v in enumerate(chromosome_state):
#             if int(v) == 1:
#                 weight_sum += X[i + 1][0]
#                 value_sum += X[i + 1][1]
#         fitnesses.append([value_sum, weight_sum])
#     return fitnesses
#
# # 筛选
# def filter(chromosome_states, fitnesses):
#     # 重量大于80的淘汰
#     index = len(fitnesses) - 1
#     while index >= 0:
#         index -= 1
#         if fitnesses[index][1] > WEIGHT_LIMIT:
#             chromosome_states.pop(index)
#             fitnesses.pop(index)
#
#     # 遴选
#     selected_index = [0] * len(chromosome_states)
#     for i in range(SELECT_NUMBER):
#         j = chromosome_states.index(random.choice(chromosome_states))
#         selected_index[j] += 1
#
#     return selected_index
#
# # 产生下一代
# def crossover(chromosome_states, selected_index):
#     chromosome_states_new = []
#     index = len(chromosome_states) - 1
#     while index >= 0:
#         index -= 1
#         chromosome_state = chromosome_states.pop(index)
#         for i in range(selected_index[index]):
#             chromosome_state_x = random.choice(chromosome_states)
#             pos = random.choice(range(1, CHROMOSOME_SIZE - 1))
#             chromosome_states_new.append(chromosome_state[: pos] + chromosome_state_x[pos:])
#         chromosome_states.insert(index, chromosome_state)
#     return chromosome_states_new
#
# def runLogistic():
#     chromosome_states = init()
#     n = 100
#     while n > 0:
#         n -= 1
#         # 适应度计算
#         fitnesses = fitness(chromosome_states)
#         print(fitnesses)
#         if is_finished(fitnesses):
#             break
#
#         # 遴选
#         selected_index = filter(chromosome_states, fitnesses)
#         chromosome_states = crossover(chromosome_states, selected_index)
#
# runLogistic()
#
#

'''
本案例用来测试 遗传算法: 使用遗传算法解决背包问题
参考文档: https://blog.csdn.net/BigDeng_2014/article/details/78043232
'''

import os
import random
from copy import deepcopy

class GAType(): # 种群32个
    def __init__(self, obj_count):
        self.gene = [0 for x in range(0, obj_count, 1)]   # 序列编码 0 / 1
        self.fitness = 0 # 适应度
        self.cho_feq = 0 # choose 选择概率
        self.cum_feq = 0 # cumulative 累积概率

class genetic():
    def __init__(self, value, weight, max_weight):
        self.value = value
        self.weight = weight
        self.max_weight = max_weight
        self.obj_count = len(weight)
        self._gatype = [GAType(self.obj_count) for x in range(0, population_size, 1)]  # 初始化32个种群
        self.total_fitness = 0

    def avoid_zero(self):
        '''防止遗传的下一代为全零，若为全零，则将随机的位数（1-7）置1'''
        flag = 0
        for i in range(0, population_size, 1):
            res = []
            for j in range(0, self.obj_count, 1):
                res.append(self._gatype[i].gene[j])
            if [0 for x in range(0, self.obj_count, 1)] == res: # 全零
                # print('找到了全零的状态！')
                flag = 1
                set_one = random.randint(1,self.obj_count)
                for k in range(0,set_one,1):
                    idx = random.randint(0,self.obj_count-1)
                    self._gatype[i].gene[idx] = 1
                # print(self._gatype[i].gene)
        return True if flag else False

    def initialize(self):
        '''初始化种群'''
        for i in range(0, population_size, 1):
            while (1): # 保证不全为零
                res = []
                for j in range(0, self.obj_count, 1):
                    self._gatype[i].gene[j] = random.randint(0,1)
                    res.append(self._gatype[i].gene[j])
                if [0 for x in range(0, self.obj_count, 1)] != res:
                    break

    def envaluateFitness(self):
        '''适应度评估 = 背包内装入物品的总价值，如果超出max_weight，则置1（惩罚性措施）'''
        self.total_fitness = 0 # 每次计算时，先将总数置0
        for i in range(0, population_size, 1):
            max_w = 0
            self._gatype[i].fitness = 0  # 置0后再计算
            for j in range(0, self.obj_count, 1):
                if self._gatype[i].gene[j] == 1:
                    self._gatype[i].fitness += self.value[j] # 适应度
                    max_w += self.weight[j] # 最大重量限制
            if max_w > self.max_weight:
                self._gatype[i].fitness = 1 # 惩罚性措施
            if 0 == self._gatype[i].fitness: # 出现了全零的种群
                self.avoid_zero()
                i = i - 1  # 重新计算该种群的fitness
            else:
                self.total_fitness += self._gatype[i].fitness # 种群的所有适应度
        if 0 == self.total_fitness:
            print('total_fitness = 0 ')

    def select(self):
        '''采用选择概率和累积概率来做选择，得出下一代种群（个数不变）
        对环境适应度高的个体，后代多，反之后代少，最后只剩下强者'''
        last_cho_feq = 0
        for i in range(0, population_size, 1):
            try:
                self._gatype[i].cho_feq = self._gatype[i].fitness / float(self.total_fitness) # 选择概率
            except:
                # print('error', self.total_fitness)
                pass
            self._gatype[i].cum_feq = last_cho_feq + self._gatype[i].cho_feq # 累积概率
            last_cho_feq = self._gatype[i].cum_feq

        # _next = deepcopy(self._gatype)  # 下一代种群，参与到后续的交叉和变异
        _next = [GAType(self.obj_count) for x in range(0, population_size, 1)]
        for i in range(0, population_size, 1):
            choose_standard = random.randint(1, 100) / 100.0
            # print('choose_standard: %f' % choose_standard)
            if choose_standard < self._gatype[0].cum_feq: # 选出下一代种群
                _next[i] = self._gatype[0]
            else:
                for j in range(1, population_size, 1):
                    if self._gatype[j-1].cum_feq <= choose_standard < self._gatype[j].cum_feq:
                        _next[i] = self._gatype[j]
        self._gatype = deepcopy(_next)
        self.avoid_zero() # 全零是不可避免的？？

    def crossover(self):
        '''采用交叉概率p_cross进行控制，从所有种群中，选择出两个种群，进行交叉互换'''
        first = -1
        for i in range(0, population_size, 1):
            choose_standard = random.randint(1, 100) / 100.0
            if choose_standard <= p_cross: # 选出两个需要交叉的种群
                if first < 0:
                    first = i
                else:
                    self.exchangeOver(first, i)
                    first = -1

    def exchangeOver(self,first,second):
        '''交叉互换'''
        exchange_num = random.randint(1, self.obj_count) # 需要交换的位置数量
        # print(exchange_num)
        for i in range(0, exchange_num, 1):
            idx = random.randint(0, self.obj_count - 1)
            self._gatype[first].gene[idx], self._gatype[second].gene[idx] = \
            self._gatype[second].gene[idx], self._gatype[first].gene[idx]

    def mutation(self):
        '''随机数小于变异概率时，触发变异'''
        for i in range(0, population_size, 1):
            choose_standard = random.randint(1, 100) / 100.0
            if choose_standard <= p_mutation: # 选出需要变异的种群
                self.reverseGene(i)

    def reverseGene(self, index):
        '''变异，将0置1，将1置0'''
        reverse_num = random.randint(1, self.obj_count)  # 需要变异的位置数量
        for i in range(0, reverse_num, 1):
            idx = random.randint(0, self.obj_count - 1)
            self._gatype[index].gene[idx] = 1 - self._gatype[index].gene[idx]

    def genetic_result(self):
        cnt = 0
        while (1):
            cnt = cnt + 1
            if cnt > 100:
                break
            self.initialize()
            self.envaluateFitness()
            for i in range(0, max_generations, 1):
                self.select()
                self.crossover()
                self.mutation()
                self.envaluateFitness()
            if True == self.is_optimal_solution(self._gatype, opt_result):
                print('循环的次数为：%d' % cnt)
                break
        self.show_res(self._gatype)

    def is_optimal_solution(self, gatype0, opt_result):
        '''判断是否达到了最优解'''
        for i in range(0, population_size, 1):
            res_list = []
            for j in range(0, self.obj_count, 1):
                res_list.append(gatype0[i].gene[j])
            if opt_result == res_list:
                return True
        return False

    def show_res(self, gatype0):
        '''显示所有种群的取值'''
        res = []
        res_list = []
        for i in range(0, population_size, 1):
            print('种群：%d --- ' % i),
            list0 = []
            for j in range(0, self.obj_count, 1):
                list0.append(gatype0[i].gene[j])
                print(gatype0[i].gene[j]),
            print(' --- 总价值：%d' % gatype0[i].fitness)
            res.append(gatype0[i].fitness)
            res_list.append(list0)
        #
        max_index = 0
        for i in range(0, len(res), 1):
            if res[max_index] < res[i]:
                max_index = i
        weight_all = 0
        for j in range(0, self.obj_count, 1):
            if gatype0[max_index].gene[j] == 1:
                weight_all += self.weight[j]
        print('当前算法的最优解(种群%d):' % max_index),
        print(res_list[max_index]),
        print('总重量（不超过%d）：' % self.max_weight),
        print(weight_all),
        print('总价值：'),
        print(res[max_index])

if __name__ == '__main__':
    weight = [35,30,60,50,40,10,25]
    value  = [10,40,30,50,35,40,30]
    max_weight = 150
    opt_result = [1,1,0,1,0,1,1]  # 全局最优解：[1,2,4,6,7] - [35,30,50,10,25] = 150 [10,40,50,40,30] = 170

    population_size = 32  # 种群
    max_generations = 500  # 进化代数
    p_cross = 0.8  # 交叉概率
    p_mutation = 0.15  # 变异概率

    genetic(value, weight, max_weight).genetic_result()
