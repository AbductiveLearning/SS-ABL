from sklearn import linear_model
import numpy as np
import json
from itertools import combinations

is_debug = False

def debug_print(*args):
    if is_debug:
        print(*args)

class SentenceModel:
    def __init__(self):
        self.LARGE = 1000
        self.HUGE = 30000
        self.EXTRA_HUGE = 300000

        self.baseline_model = linear_model.LinearRegression()               
        self.rate_model = linear_model.LinearRegression(fit_intercept=False)
         
    def fit_baseline(self, moneys, months):
        self.baseline_model.fit(moneys, months)
    
    def predict_baseline(self, moneys):
        return self.baseline_model.predict(moneys)

    def fit_rate(self, moneys, attrs, months):
        rates = []
        for money, month in zip(moneys, months):
            baseline = self.predict_baseline([money])
            rates.append((month - baseline) / baseline) 
        self.rate_model.fit(attrs, rates)
    
    def predict_rate(self, attrs):
        return self.rate_model.predict(attrs)
    
    def predict(self, moneys, attrs):
        baseline = self.predict_baseline(moneys)
        rate = self.predict_rate(attrs).flatten()
        months_hat = np.multiply(baseline, (rate + 1))
        return months_hat
    
    def test(self, moneys, attrs, months, filenames_test = None, ahs_test = None):
        assert len(moneys) == len(attrs)
        assert len(attrs) == len(months)
        test_num = float(len(moneys))

        err_less_month = [1, 2, 3, 6]
        err_less_than_num = [0] * (max(err_less_month) + 1)
        perc_cnt = [0] * 11
        percent = 0
        mae = 0
        mse = 0
        
        for idx, (money, attr, month) in enumerate(zip(moneys, attrs, months)):
            month_hat = self.predict([money], [attr])
            
            if abs(month - month_hat) > 5:
                if filenames_test != None:
                    debug_print("\nFilename:", filenames_test[idx])

                if ahs_test != None:
                    debug_print("ah:", ahs_test[idx])

                debug_print("Money:", money)
                debug_print("Attrs:", attr)
                debug_print("Predict:", month_hat)
                debug_print("Actual:", month)
            
            absolutly_error = abs(month - month_hat)
            for month in err_less_month:
                if absolutly_error <= month:
                    err_less_than_num[month] += 1

            mae += (absolutly_error) / test_num
            mse += (absolutly_error * absolutly_error) / test_num

            percentage = absolutly_error / month
            percent += (percentage) / test_num
            perc_cnt[min(int(percentage * 100 / 5), 10)] += 1
            
        for month in err_less_month:
            print("Error <= %d month percentage:" % (month), err_less_than_num[month] / test_num * 100)

        print('MSE：', mse)
        print('MAE：', mae)
        print('Average error percent: ', percent)
        print('Percentage distribution: ', perc_cnt)
        return mae, mse, percent, perc_cnt
    
    def fit(self, moneys, attrs, labels, times = 1):
        punish_adjust = np.array([0] * len(moneys))
        for j in range(times):
            self.fit_baseline(moneys, np.array(labels) - punish_adjust)
            self.fit_rate(moneys, attrs, labels)

            baseline = self.predict_baseline(moneys)
            rate = self.predict_rate(attrs).flatten()
            punish_adjust = np.multiply(baseline, rate)

    def show_param(self):
        print("Baseline predictor")
        print(self.baseline_model.coef_)
        print(self.baseline_model.intercept_)
        print("Adapt rate predictor")
        print(self.rate_model.coef_)
        print(self.rate_model.intercept_)

    def get_param(self):
        baseline_param = "coef: " + ", ".join(str(d) for d in self.baseline_model.coef_) + " intercept: " + str(self.baseline_model.intercept_)
        rate_param = "coef: " + ", ".join(str(d) for d in self.rate_model.coef_) + " intercept: " + str(self.rate_model.intercept_)
        return baseline_param, rate_param

    
    def threshold_judge(self, pred, y, threshold, case):
        if case == 0:
            return abs(pred - y) / y < threshold
        if case > 0:
            return (pred - y) / y >= threshold
        if case < 0:
            return (y - pred) / y >= threshold

    def split_data(self, moneys, attrs, months, filenames = None, ahs = None, threshold = 0.2):
        ret_moneys = [[] for _ in range(3)]
        ret_months = [[] for _ in range(3)]
        ret_attrs = [[] for _ in range(3)]
        ret_filenames = [[] for _ in range(3)]
        ret_ahs = [[] for _ in range(3)]
        ret_errors = [[] for _ in range(3)]
        
        for idx, (money, attr, month) in enumerate(zip(moneys, attrs, months)):
            month_hat = self.predict([money], [attr])
            for case in range(3):
                if self.threshold_judge(month_hat, month, threshold, case - 1):
                    ret_moneys[case].append(money)
                    ret_months[case].append(month)
                    ret_attrs[case].append(attr)
                    ret_filenames[case].append(filenames[idx])
                    ret_ahs[case].append(ahs[idx])
                    ret_errors[case].append((abs(month - month_hat) / month, month, money, attr, month_hat))

        return ret_moneys, ret_attrs, ret_months, ret_filenames, ret_ahs, ret_errors
