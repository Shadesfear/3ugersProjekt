x0 = [0.0, 0.0]
def func(x,a,b):
	return a * x * x + b

fit = optimization.curve_fit(func, df2.x, df2.y, x0, sigma = df2.sigma)
ap = fit[0].tolist()[0]
bp = fit[0].tolist()[1]