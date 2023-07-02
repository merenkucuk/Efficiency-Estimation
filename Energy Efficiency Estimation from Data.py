import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# KNN Regression Implementation
# There are 2 distance ways in wikipedia euclidean and mahalanobis distance.
# We implement both but use euclidean due to take better results.
# You can change load data with colName variable.
class KNN_Regression:
    def __init__(self, k=1):
        self.k = k
    # Predict function with weighted or not
    def predict(self, val_te, x_tr, y_tr, weighted=False):
        if weighted:
            predicted_vals = []
            for k in val_te:
                # Distance to calculate shortest between the 2 points irrespective of the dimensions.
                distances = [self.euclidean_distance(k, x_train, 2) for x_train in x_tr]

                k_indexes = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(distances))[:self.k]]

                k_nearest_vals = []
                for i in range(len(k_indexes)):
                    k_nearest_vals.append(y_tr[k_indexes[i]])

                k_distances = []
                for i in range(len(k_indexes)):
                    k_distances.append(distances[k_indexes[i]])

                weighted_dict = {}

                for i in range(len(k_distances)):
                    k_distances[i] = k_distances[i] + 2

                for i in range(len(k_nearest_vals)):
                    if k_nearest_vals[i] in weighted_dict:
                        weighted_dict[k_nearest_vals[i]] += (1 / k_distances[i])
                    else:
                        weighted_dict[k_nearest_vals[i]] = (1 / k_distances[i])

                val = 0
                divided = 0
                for i in weighted_dict.keys():
                    val += i * weighted_dict[i]
                    divided += weighted_dict[i]

                result = val / divided
                predicted_vals.append(result)

            return np.array(predicted_vals)
        else:
            predicted_vals = []
            for k in val_te:
                distances = [self.euclidean_distance(k, x_train, 2) for x_train in x_tr]

                k_indexes = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(distances))[:self.k]]

                k_nearest_vals = []

                for i in range(len(k_indexes)):
                    k_nearest_vals.append(y_tr[k_indexes[i]])

                avr_val = 0
                for i in range(len(k_nearest_vals)):
                    avr_val += k_nearest_vals[i]

                avr_val = avr_val / len(k_nearest_vals)
                predicted_vals.append(avr_val)

            return np.array(predicted_vals)

    def euclidean_distance(self, row1, row2, p=2):
        distance = 0.0
        for i in range(len(row1) - 1):
            distance += (row1[i] - row2[i]) ** p

        return distance ** (1 / p)

    def mahalanobis_distance(y=None, data=None, cov=None):
        y_mu = y - np.mean(data)

        if not cov:
            cov = np.cov(data.values.T)

        inv_covmat = np.linalg.inv(cov)
        left = np.dot(y_mu, inv_covmat)
        mahal = np.dot(left, y_mu.T)

        return mahal.diagonal()


def get_min_max(x, min, max):
    mins = []  # stores minimum values of each columns respectively
    maxs = []  # stores maximum values of each columns respectively
    for column_index in range(x.shape[1]):
        min_value = float("inf")
        max_value = float("-inf")

        for row_index in range(x.shape[0]):
            if x[row_index, column_index] < min_value:
                min_value = x[row_index, column_index]
            elif x[row_index, column_index] > max_value:
                max_value = x[row_index, column_index]

        mins.append(min_value)
        maxs.append(max_value)

    min_x = np.array(mins)
    max_x = np.array(maxs)

    if (min == True) and (max == False):
        return min_x

    elif (min == False) and (max == True):
        return max_x


def normalization(data):
    normalized_data = np.zeros(data.shape)
    min_x, max_x = get_min_max(data, True, False), get_min_max(data, False, True)

    for row_number in range(data.shape[0]):
        for column_number in range(data.shape[1]):
            normalized_data[row_number, column_number] = (data[row_number, column_number] - min_x[column_number]) / (max_x[column_number] - min_x[column_number])

    return normalized_data


def meanAbsoluteError(actual, predicted):
    N = actual.shape[0]
    mae_val = np.sum(np.absolute(predicted - actual))
    mae_val /= N

    return mae_val


def train_test_split(data, split, trainingSet, testSet):
    for x in range(len(data)):
        if np.random.random() < split:
            trainingSet.append(data[x])
        else:
            testSet.append(data[x])



def read_drop_shuffle_parse(colName="Cooling_Load"):
    # reading csv file with pandas
    df = pd.read_csv('energy_efficiency_data.csv')
    # drop column
    df.drop(colName, axis=1)
    # data to numpy array
    df = df.to_numpy()
    # we use random.shuffle due to you said allow it on 3rd week lab lecture
    np.random.shuffle(df)
    return df[:, :-1], df[:, -1], df


def cross_validation():
    validate = int(len(df) * i / cross_value)
    cross_size_value = int(len(df) / cross_value)
    if i == (cross_value - 1):
        x_val = features[validate:, :]
        x_normalized_val = normalized_features[validate:, :]
        y_val = labels[validate:]
    else:
        x_val = features[validate:validate + cross_size_value, :]
        x_normalized_val = normalized_features[validate:validate + cross_size_value, :]
        y_val = labels[validate:validate + cross_size_value]
    x_tr = features[validate + cross_size_value:, :]
    x_normalized_tr = normalized_features[validate + cross_size_value:, :]
    y_tr = labels[validate + cross_size_value:]
    if i != 0:
        x_train2 = features[:validate, :]
        x_normalized_train2 = normalized_features[:validate, :]
        y_train2 = labels[:validate]

        x_tr = np.concatenate((x_tr, x_train2), axis=0)
        x_normalized_tr = np.concatenate((x_normalized_tr, x_normalized_train2), axis=0)
        y_tr = np.concatenate((y_tr, y_train2), axis=0)
    return x_val, x_normalized_val, y_val, x_tr, x_normalized_tr, y_tr


# trainingSet = []
# testSet = []
# split = 0.6
# train_test_split(df, split, trainingSet, testSet)
# trainingSet = pd.DataFrame(trainingSet)
# testSet = pd.DataFrame(testSet)
# X_tr = trainingSet.drop([8,9], axis=1)
# y_tr = trainingSet[8]
# test = testSet.sort_values(8)
# X_te = test.drop([8, 9], axis=1)
# y_te = test[8]
# 8 for Heating 9 for Cooling
# here is the train and test split but not permission to use dataframe
# we do not use and make it another way  but implement it anyway

# colName = "Cooling_Load"
# colName = "Heating_Load"
colName = "Heating_Load"
features, labels, df = read_drop_shuffle_parse(colName)

# normalize data
normalized_features = normalization(features)
avg_mae_uw_un = 0
avg_mae_uw_n = 0
avg_mae_w_un = 0
avg_mae_w_n = 0
cross_value = 5
list_to_plot = []
list_to_plot1 = []
# k - value list (1,3,5,7,9)
# k (5) cross validation
for k in range(1, 11, 2):
    for i in range(cross_value):
        x_validate, x_normalized_validate, y_validate, x_train, x_normalized_train, y_train = cross_validation()

        # Predict Not Weighted - Not Normalized
        model_uw_un = KNN_Regression(k)
        prediction_uw_un = model_uw_un.predict(x_validate, x_train, y_train)
        mae_uw_un = meanAbsoluteError(y_validate, prediction_uw_un)
        avg_mae_uw_un += mae_uw_un

        # Predict Not Weighted - Normalized
        model_uw_n = KNN_Regression(k)
        predictions_uw_n = model_uw_n.predict(x_normalized_validate, x_normalized_train, y_train)
        mae_uw_n = meanAbsoluteError(y_validate, predictions_uw_n)
        avg_mae_uw_n += mae_uw_n

        # Predict Weighted - Not Normalized
        model_w_un = KNN_Regression(k)
        prediction_w_un = model_w_un.predict(x_validate, x_train, y_train, True)
        mae_w_un = meanAbsoluteError(y_validate, prediction_w_un)
        avg_mae_w_un += mae_w_un

        # Predict Weighted - Normalized
        model_w_n = KNN_Regression(k)
        prediction_w_n = model_w_n.predict(x_normalized_validate, x_normalized_train, y_train, True)
        mae_w_n = meanAbsoluteError(y_validate, prediction_w_n)
        avg_mae_w_n += mae_w_n

        list_to_plot.append(mae_uw_n)
        list_to_plot.append(mae_uw_un)
        list_to_plot.append(mae_w_n)
        list_to_plot.append(mae_w_un)

        if i == (cross_value - 1):
            list_to_plot1.append(avg_mae_uw_n / 5)
            list_to_plot1.append(avg_mae_uw_un / 5)
            list_to_plot1.append(avg_mae_w_n / 5)
            list_to_plot1.append(avg_mae_w_un / 5)
            avg_mae_w_n = 0
            avg_mae_w_un = 0
            avg_mae_uw_n = 0
            avg_mae_uw_un = 0


def plot_table(k: int):
    k1_mae_list = list_to_plot[0:20]
    k2_mae_list = list_to_plot[20:40]
    k3_mae_list = list_to_plot[40:60]
    k4_mae_list = list_to_plot[60:80]
    k5_mae_list = list_to_plot[80:]
    k1_average_list = list_to_plot1[0:4]
    k2_average_list = list_to_plot1[4:8]
    k3_average_list = list_to_plot1[8:12]
    k4_average_list = list_to_plot1[12:16]
    k5_average_list = list_to_plot1[16:20]
    mae1list = []
    mae2list = []
    mae3list = []
    mae4list = []
    mae5list = []
    data = []
    if k == 1:
        for i in range(0, 4):
            mae1list.append(k1_mae_list[i])
            mae2list.append(k1_mae_list[i + 4])
            mae3list.append(k1_mae_list[i + 8])
            mae4list.append(k1_mae_list[i + 12])
            mae5list.append(k1_mae_list[i + 16])
        data = [k1_average_list, mae5list, mae4list, mae3list, mae2list, mae1list]
    elif k == 3:
        for i in range(0, 4):
            mae1list.append(k2_mae_list[i])
            mae2list.append(k2_mae_list[i + 4])
            mae3list.append(k2_mae_list[i + 8])
            mae4list.append(k2_mae_list[i + 12])
            mae5list.append(k2_mae_list[i + 16])
        data = [k2_average_list, mae5list, mae4list, mae3list, mae2list, mae1list]
    elif k == 5:
        for i in range(0, 4):
            mae1list.append(k3_mae_list[i])
            mae2list.append(k3_mae_list[i + 4])
            mae3list.append(k3_mae_list[i + 8])
            mae4list.append(k3_mae_list[i + 12])
            mae5list.append(k3_mae_list[i + 16])
        data = [k3_average_list, mae5list, mae4list, mae3list, mae2list, mae1list]
    elif k == 7:
        for i in range(0, 4):
            mae1list.append(k4_mae_list[i])
            mae2list.append(k4_mae_list[i + 4])
            mae3list.append(k4_mae_list[i + 8])
            mae4list.append(k4_mae_list[i + 12])
            mae5list.append(k4_mae_list[i + 16])
        data = [k4_average_list, mae5list, mae4list, mae3list, mae2list, mae1list]
    elif k == 9:
        for i in range(0, 4):
            mae1list.append(k5_mae_list[i])
            mae2list.append(k5_mae_list[i + 4])
            mae3list.append(k5_mae_list[i + 8])
            mae4list.append(k5_mae_list[i + 12])
            mae5list.append(k5_mae_list[i + 16])
        data = [k5_average_list, mae5list, mae4list, mae3list, mae2list, mae1list]
    else:
        print("k value should be 1, 3, 5, 7 or 9")
    print("Plotting table is success for k = " + str(k))
    columns = ('Not Weighted - Not Normalized', 'Not Weighted - Normalized', 'Weighted - Not Normalized', "Weighted - Normalized")
    rows = ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5", "Average MAE"]

    # Get some pastel shades for the colors
    colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))
    n_rows = len(data)

    index = np.arange(len(columns)) + 0.3

    cell_text = []
    for row in range(n_rows):
        plt.plot(index, data[row], color=colors[row])
        y_offset = data[row]
        cell_text.append([x for x in y_offset])
    colors = colors[::-1]
    cell_text.reverse()
    the_table = plt.table(cellText=cell_text,
                          colLabels=columns,
                          rowLabels=rows,
                          rowColours=colors,
                          loc='bottom')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(8)
    plt.subplots_adjust(left=0.25, bottom=0.25)

    plt.ylabel("Mean Absolute Error".format("value_increment"))
    plt.xticks([])
    if colName == "Heating_Load":
        plt.title('Cooling_Load Mean Absolute Error for k = ' + str(k))
    else:
        plt.title('Heating_Load Mean Absolute Error for k = ' + str(k))


    plt.show()

    return "Plotting table is success for k = " + str(k)


# k = 1 table
plot_table(1)
# k = 3 table
plot_table(3)
# k = 5 table
plot_table(5)
# k = 7 table
plot_table(7)
# k = 9 table
plot_table(9)
