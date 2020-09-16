import copy
import numpy as np

def parse_condition(rule):
    ret = []
    for r in rule.split("^"):
        d = r.split(':')
        ret.append((int(d[0]), int(d[1])))
    return ret

class CheckRules():
    def __init__(self, filename):
        self.rule_list = []
        if filename is None:
            return
        with open(filename) as fin:
            for rule in fin:
                rule = rule.strip().replace(" ", "").split('#')
                left, right, p = rule
                left = parse_condition(left)
                right = parse_condition(right)
                self.rule_list.append((left, right, float(p)))

    def __satisfied(self, status, rule):
        for r in rule:
            if status[r[0]] != r[1]:
                return False
        return True

    def fix(self, status):
        ret = []
        for rule in self.rule_list:
            left, right, p = rule
            if (not self.__satisfied(status, left)):
                continue
            new_status = copy.deepcopy(status)
            for r in right:
                new_status[r[0]] = r[1]
            ret.append((new_status, p))
        return ret

    def judge(self, status):
        total_p = 1
        for rule in self.rule_list:
            left, right, p = rule
            if self.__satisfied(status, left):
                if self.__satisfied(status, right):
                    total_p *= p
                else:
                    total_p *= (1 - p)
            if total_p < 1e-6:
                return total_p
        return total_p

if __name__ == "__main__":
    check = CheckRules("rule_file.txt")
    status = np.array([1,1,1,0,0,1,1,0,0])
    print(check.judge(status))
