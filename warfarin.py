import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
from sklearn import linear_model
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()
parser.add_argument('--baseline', type=str, default='', help='Which baseline? [fixed, clinical, linear, lasso]')
parser.add_argument('--lr', type=float, default=0.1, help='Learning rate for linear baseline')
parser.add_argument('--lambda-2', type=float, default=0.1, help='Lambda value if using lasso')
parser.add_argument('--impute', action='store_true', help='Set to impute data.')
parser.add_argument('--graph', action='store_true', help='Set to graph data')
parser.add_argument('--n-samples', type=int, default=10, help='Number of samples')
parser.add_argument('--use-alternative-rewards', action='store_true')

def convert_age_range_to_number(age):
	if age == '0 - 9':
		return 0
	elif age == '10 - 19':
		return 1
	elif age == '20 - 29':
		return 2
	elif age == '30 - 39':
		return 3
	elif age == '40 - 49':
		return 4
	elif age == '50 - 59':
		return 5
	elif age == '60 - 69':
		return 6
	elif age == '70 - 79':
		return 7
	elif age == '80 - 89':
		return 8
	else:
		return 9

def bin_dosage(d):
	if d < 21:
		return 0
	elif d >= 21 and d <= 49:
		return 1
	else:
		return 2

def load_warfarin_data(args):
	warfarin_frame = pd.read_csv('data/warfarin.csv')
	print(len(warfarin_frame))
	# drop patients with no labels
	if args.impute:
		print('Imputing')
		warfarin_frame['Carbamazepine (Tegretol)'].fillna(0, inplace=True)
		warfarin_frame['Phenytoin (Dilantin)'].fillna(0, inplace=True)
		warfarin_frame['Rifampin or Rifampicin'].fillna(0, inplace=True)
		warfarin_frame['Amiodarone (Cordarone)'].fillna(0, inplace=True)
	if args.baseline == 'clinical' or args.baseline == 'linear' or args.baseline == 'lasso':
		warfarin_frame = warfarin_frame.dropna(subset=['Therapeutic Dose of Warfarin', 'Age', 'Weight (kg)', 'Height (cm)', 'Race', 'Carbamazepine (Tegretol)', 'Phenytoin (Dilantin)', 'Rifampin or Rifampicin', 'Amiodarone (Cordarone)'])
	else:
		warfarin_frame = warfarin_frame.dropna(subset=['Therapeutic Dose of Warfarin'])
		print(len(warfarin_frame))
	warfarin_frame = warfarin_frame.reindex(np.random.permutation(warfarin_frame.index))
	labels = warfarin_frame['Therapeutic Dose of Warfarin']
	features = warfarin_frame.drop('Therapeutic Dose of Warfarin', axis=1)
	features['Age (number)'] = features['Age'].transform(convert_age_range_to_number)
	return features, labels

def load_warfarin_data_categorical(warfarin_frame):
	warfarin_frame['Age'] = warfarin_frame['Age'].astype('category')
	warfarin_frame['Race'] = warfarin_frame['Race'].astype('category')
	warfarin_frame['Ethnicity'] = warfarin_frame['Ethnicity'].astype('category')
	warfarin_frame['Indication for Warfarin Treatment'] = warfarin_frame['Indication for Warfarin Treatment'].astype('category')
	warfarin_frame['Carbamazepine (Tegretol)'] = warfarin_frame['Carbamazepine (Tegretol)'].astype('category')
	warfarin_frame['Phenytoin (Dilantin)'] = warfarin_frame['Phenytoin (Dilantin)'].astype('category')
	warfarin_frame['Rifampin or Rifampicin'] = warfarin_frame['Rifampin or Rifampicin'].astype('category')
	warfarin_frame['Amiodarone (Cordarone)'] = warfarin_frame['Amiodarone (Cordarone)'].astype('category')
	warfarin_frame['Gender'] = warfarin_frame['Gender'].astype('category')
	warfarin_frame[warfarin_frame.select_dtypes(['category']).columns] = warfarin_frame[warfarin_frame.select_dtypes(['category']).columns].apply(lambda x: x.cat.codes)
	return warfarin_frame

def convert_to_feature_matrix(warfarin_frame):
	m = []
	one_hot_age = pd.get_dummies(warfarin_frame['Age']).values.tolist()
	one_hot_race = pd.get_dummies(warfarin_frame['Race']).values.tolist()
	height = warfarin_frame['Height (cm)'].values.tolist()
	weight = warfarin_frame['Weight (kg)'].values.tolist()
	one_hot_ethnicity = pd.get_dummies(warfarin_frame['Ethnicity']).values.tolist()
	one_hot_indication = pd.get_dummies(warfarin_frame['Indication for Warfarin Treatment']).values.tolist()
	one_hot_cb = pd.get_dummies(warfarin_frame['Carbamazepine (Tegretol)']).values.tolist()
	one_hot_ph = pd.get_dummies(warfarin_frame['Phenytoin (Dilantin)']).values.tolist()
	one_hot_ri = pd.get_dummies(warfarin_frame['Rifampin or Rifampicin']).values.tolist()
	one_hot_am = pd.get_dummies(warfarin_frame['Amiodarone (Cordarone)']).values.tolist()
	one_hot_gender = pd.get_dummies(warfarin_frame['Gender']).values.tolist()
	# raise Exception()
	for i, age_row in enumerate(one_hot_age):
		r = []
		r.append(height[i])
		r.append(weight[i])
		r.extend(age_row)
		r.extend(one_hot_race[i])
		r.extend(one_hot_ethnicity[i])
		r.extend(one_hot_indication[i])
		r.extend(one_hot_cb[i])
		r.extend(one_hot_ph[i])
		r.extend(one_hot_ri[i])
		r.extend(one_hot_am[i])
		r.extend(one_hot_gender[i])
		# add bias
		r.append(1)
		m.append(r)
	mat = np.array(m)
	print(mat.shape)
	return mat

def get_reward(action, label):
	if action == label:
		return 0
	else:
		return -1

def get_regret(action, label):
	if action == label:
		return 0
	else:
		return 0 - get_reward(action, label)

def get_reward_alternative(action, label):
	if action == label:
		return 10
	if label == 2 and action != label:
		return -10
	elif label == 1 and action > label:
		return -5
	elif label == 1 and action < label:
		return 5
	elif label == 0 and action == 2:
		return -10
	else:
		return 0

def get_regret_alternative(action, label):
	if action == label:
		return 0
	if label == 2 and action != label:
		return 10 - (-10)
	elif label == 1 and action > label:
		return 10 - (-5)
	elif label == 1 and action < label:
		return 5
	elif label == 0 and action == 2:
		return 10 - (-10)
	else:
		return 0

def evaluate(labels, decisions, args):
	correct, total = 0, 0
	regrets = []
	last_10 = []
	frac_correct = []
	cur_regret = 0
	for i, label in enumerate(labels):
		label_bin = bin_dosage(label)
		if decisions[i] == label_bin:
			correct += 1
		if args.use_alternative_rewards:
			regret = get_regret_alternative(decisions[i], label_bin)
		else:
			regret = get_regret(decisions[i], label_bin)
		cur_regret += regret
		regrets.append(cur_regret)
		total += 1
		if len(last_10) == 50:
			last_10 = last_10[1:]
		if decisions[i] == label_bin:
			last_10.append(1)
		else:
			last_10.append(0)
		frac_correct.append(sum(last_10) / 50)
	return float(correct) / total, regrets, frac_correct

class Policy():
	def __init__(self):
		pass
	def choose_arm(self, feature):
		pass
	def update(self):
		pass
class FixedDosePolicy(Policy):
	def __init__(self, dose):
		self.dose = dose
	def choose_arm(self, feature):
		return bin_dosage(self.dose)
	def update(self):
		pass

#implement linUCB

class LinearBandit(nn.Module):
	def __init__(self, feature_size):
		self.linear = nn.Linear(feature_size, 3, bias=True)
	def forward(self, x):
		return self.linear(x)

class LinearPolicy(Policy):
	def __init__(self, feature_size):
		self.feature_size = feature_size
		self.model = LinearBandit(feature_size)
		self.loss = nn.CrossEntropyLoss()
		self.optimizer = nn.SGD(self.model.parameters(), args.lr)
	def choose_arm(self, feature):
		return torch.argmax(self.model(feature), dim=1)
	def update(self, feature, label):
		l = self.loss(0, self.choose_arm(feature))
		self.optimizer.zero_grad()
		l.backward()
		self.optimizer.step()

def compute_ucb(theta, A, row, alpha):
	return np.dot(theta, row) + alpha * np.sqrt(np.dot(row.T, np.dot(np.linalg.inv(A), row)))

def lin_ucb(args):
	features, labels = load_warfarin_data(args)
	labels = labels.tolist()
	features = load_warfarin_data_categorical(features)
	features = convert_to_feature_matrix(features)
	feature_size = features.shape[1]
	beta = 0.2
	# alpha = np.sqrt(0.5 * np.log((2 * features.shape[0] * 3) / beta))
	alpha = 2
	ucb_params = {'A' : [np.identity(feature_size), np.identity(feature_size), np.identity(feature_size)], 'b' : [np.zeros(feature_size), np.zeros(feature_size), np.zeros(feature_size)]}
	i = 0
	decisions = []
	for row in features:
		theta_t_0 = np.dot(np.linalg.inv(ucb_params['A'][0]), ucb_params['b'][0])
		theta_t_1 = np.dot(np.linalg.inv(ucb_params['A'][1]), ucb_params['b'][1])
		theta_t_2 = np.dot(np.linalg.inv(ucb_params['A'][2]), ucb_params['b'][2])
		p_0 = compute_ucb(theta_t_0, ucb_params['A'][0], row, alpha)
		p_1 = compute_ucb(theta_t_1, ucb_params['A'][1], row, alpha)
		p_2 = compute_ucb(theta_t_2, ucb_params['A'][2], row, alpha)
		action = np.argmax([p_0, p_1, p_2])
		payout = 1 if action == bin_dosage(labels[i]) else 0
		if args.use_alternative_rewards:
			payout = get_reward_alternative(action, labels[i])
		decisions.append(action)
		ucb_params['A'][action] = ucb_params['A'][action] + np.dot(row, row.T)
		ucb_params['b'][action] = ucb_params['b'][action] + row * payout
		i += 1
	print(decisions)
	return evaluate(labels, decisions, args)

def action_in_sets(sets, action):
	for i in range(len(sets)):
		if action in sets[i]:
			return i
	return None

def lasso_bandit(args):
	features, labels = load_warfarin_data(args)
	labels = labels.tolist()
	features = load_warfarin_data_categorical(features)
	# DIVIDE LAMBDAS BY 2
	features = convert_to_feature_matrix(features)
	feature_size = features.shape[1]
	params = {'T_it' : [[[], []] for _ in range(3)], 'S' : [[[], []] for _ in range(3)], 'lambda_1' : args.lambda_2, 'lambda_2' : args.lambda_2, 'q' : 15, 'h' : 2}
	params['B_forced'] = [np.zeros(feature_size) for i in range(3)]
	params['B_all'] = [np.zeros(feature_size) for i in range(3)]
	T_i = [set(), set(), set()]
	n = 0
	while n < 1000:
		for i, s in enumerate(T_i):
			arm_idx = i+1
			j = params['q'] * (arm_idx - 1) + 1
			while j <= params['q'] * arm_idx:
				T_i[i].add(((2 ** n)- 1) * (3 * params['q']) + j)
				j += 1
		n += 1
	decisions = []
	for t in range(features.shape[0]):
		X_t = features[t]
		forced_action = action_in_sets(T_i, t)
		if forced_action:
			params['T_it'][forced_action][0].append(X_t)
			params['S'][forced_action][0].append(X_t)
			if args.use_alternative_rewards:
				reward = get_reward_alternative(forced_action, labels[t])
			else:
				reward = get_reward(forced_action, labels[t])
			params['T_it'][forced_action][1].append(reward)
			params['S'][forced_action][1].append(reward)
			train_mat = np.stack(params['T_it'][forced_action][0])
			train_labels = np.array(params['T_it'][forced_action][1])
			params['B_forced'][forced_action] = linear_model.Lasso(params['lambda_1'], fit_intercept=False, tol=0.001).fit(train_mat, train_labels).coef_
			decisions.append(forced_action)
		else:
			K_hat = []
			for i in range(3):
				K_hat.append(np.dot(X_t, params['B_forced'][i]))
			max_K_hat = np.amax(K_hat)
			K_set = set()
			for i, k in enumerate(K_hat):
				if k >= max_K_hat - (params['h']/2):
					K_set.add(k)
			actions = [np.dot(X_t, params['B_all'][c]) for c in range(3)]
			chosen_action = np.argmax(actions)
			decisions.append(chosen_action)
			params['S'][chosen_action][0].append(X_t)
			if args.use_alternative_rewards:
				reward = get_reward_alternative(chosen_action, labels[t])
			else:
				reward = get_reward(chosen_action, labels[t])
			params['S'][chosen_action][1].append(reward)
			lambda_adjustment = (np.log(t+1) + np.log(feature_size)) / (t+1)
			params['lambda_2'] = args.lambda_2 * np.sqrt(lambda_adjustment)
			train_mat = np.stack(params['S'][chosen_action][0])
			train_labels = np.array(params['S'][chosen_action][1])
			params['B_all'][chosen_action] = linear_model.Lasso(params['lambda_2'], fit_intercept=False).fit(train_mat, train_labels).coef_
	return evaluate(labels, decisions, args)

def evaluate_baseline_policy(p, args):
	# this is the fixed dose baseline
	features, labels = load_warfarin_data(args)
	decisions = []
	for i, feature in features.iterrows():
		decisions.append(p.choose_arm(feature))
	return evaluate(labels, decisions, args)
	# print('Accuracy: ', evaluate(labels, decisions))

def evaluate_linear_policy(p, args):
	features, labels = load_warfarin_data_categorical(args)
	decisions = []
	for i, feature in features.iterrows():
		arm = p.choose_arm(feature)
		decisions.append(arm)
		p.update()

class WarfarinClinicalDosingPolicy(Policy):
	def __init__(self):
		pass
	def choose_arm(self, feature):
		score = 4.0376
		score -= 0.2546 * feature['Age (number)']
		score += 0.0118 * feature['Height (cm)']
		score += 0.0134 * feature['Weight (kg)']
		# racial categories
		if feature['Race'] == 'Black or African American':
			score += 0.4060
		elif feature['Race'] == 'Asian':
			score -= 0.6752
		elif feature['Race'] == 'Unknown':
			score += 0.0443
		if feature['Carbamazepine (Tegretol)'] == 1 \
			or feature['Phenytoin (Dilantin)'] == 1 \
			or feature['Rifampin or Rifampicin'] == 1:
			score += 1.2799
		if feature['Amiodarone (Cordarone)'] == 1:
			score -= 0.5695
		dosage = score ** 2
		return bin_dosage(dosage)

def graph_data(args):
	regret_samples, perf_samples, frac_samples = [], [], []
	while len(perf_samples) != args.n_samples:
		if args.baseline == 'linear':
			performance, regret, frac = lin_ucb(args)
		elif args.baseline == 'lasso':
			performance, regret, frac = lasso_bandit(args)
		elif args.baseline == 'clinical':
			clinical_dosing = WarfarinClinicalDosingPolicy()
			performance, regret, frac = evaluate_baseline_policy(clinical_dosing, args)
		regret_samples.append(regret)
		perf_samples.append(performance)
		frac_samples.append(frac)
	print(args.baseline, ' performance average over ', args.n_samples, 'samples', np.mean(performance))
	x_axis = [t for t in range(len(regret_samples[0]))]
	time_series_df = pd.DataFrame(np.array(regret_samples))
	smooth_path = time_series_df.rolling(len(regret_samples)).mean().iloc[len(regret_samples)-1]
	path_deviation = 2 * time_series_df.rolling(len(regret_samples)).std().iloc[len(regret_samples)-1]
	_, ax = plt.subplots()
	ax.plot(smooth_path, linewidth=2)
	ax.fill_between(path_deviation.index, (smooth_path-2*path_deviation), (smooth_path+2*path_deviation), color='#539caf')
	ax.set_xlabel('Time steps')
	ax.set_ylabel('Regret')
	ax.set_title('Regret plot')
	time_series_df = pd.DataFrame(np.array(frac_samples))
	smooth_path = time_series_df.rolling(len(frac_samples)).mean().iloc[len(frac_samples)-1]
	path_deviation = 2 * time_series_df.rolling(len(frac_samples)).std().iloc[len(frac_samples)-1]
	if not args.use_alternative_rewards:
		plt.savefig('regret_' + str(args.baseline) + '.png', dpi=150)
	else:
		plt.savefig('regret_' + str(args.baseline) + '_alt.png', dpi=150)
	_, ax2 = plt.subplots()
	ax2.plot(smooth_path, linewidth=2)
	ax2.fill_between(path_deviation.index, (smooth_path-2*path_deviation), (smooth_path+2*path_deviation), color='#539caf')
	ax2.set_xlabel('Time steps')
	ax2.set_ylabel('Fraction correct')
	ax2.set_title('Fraction Correct over time')
	if not args.use_alternative_rewards:
		plt.savefig('fraction_correct_' + str(args.baseline) + '.png', dpi=150)
	else:
		plt.savefig('fraction_correct_' + str(args.baseline) + '_alt.png', dpi=150)

def main():
	args = parser.parse_args()
	fixed_dose = FixedDosePolicy(35)
	clinical_dosing = WarfarinClinicalDosingPolicy()
	if args.graph:
		graph_data(args)
	elif args.baseline == 'clinical':
		evaluate_baseline_policy(clinical_dosing, args)
	elif args.baseline == 'fixed':
		evaluate_baseline_policy(fixed_dose, args)
	elif args.baseline == 'linear':
		lin_ucb(args)
	elif args.baseline == 'lasso':
		lasso_bandit(args)
	# evaluate linear baseline

if __name__ == '__main__':
	main()
	