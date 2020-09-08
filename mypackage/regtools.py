import statsmodels.api as sm
from seaborn import regplot

def lm(x, y):
	x = sm.add_constant(x)
	model = sm.OLS(y, x).fit()
	return model
# get summary: model.summary()
# get slope and inteception: model.params 
# get R^2: model.rsquared
# get outliers: model.get_influence().summary_frame()
# get residuals: model.resid
# get sum of squared residuals: model.ssr, equal to np.sum(lm.resid ** 2)
# get intercept and slope: model.params


def find_outliers(x, y, ckd_thr=0.1):

    # Cook's distance is used to find Outliers

    model = lm(x, y)
    cooks_dist = model.get_influence().summary_frame()['cooks_d'].values
    outlier_indx = np.where(cooks_dist > ckd_thr)[0]
    return outlier_indx