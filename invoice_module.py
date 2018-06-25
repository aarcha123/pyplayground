import sys,getopt
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFE
import math
import pickle

saved_model_file = 'finalized_model.sav'
model = None
X_test = None
Y_test = None
X_train = None
Y_train = None
cust_dict = None


def main(argv):
    trainfile=''
    testfile=''
    datafile=''
    try:
        opts, args = getopt.getopt(argv, "ht:e:p", ["train=", "test=","predict="])
    except getopt.GetoptError:
        print('test.py -t <traindata> -v <validatedata> -p <predictdata>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -t <traindata> -v <validatedata> -p <predictdata>')
            sys.exit()
        elif opt in ("-t", "--train"):
            train(arg)
        elif opt in ("-e", "--test"):
            testfile = arg
        elif opt in ("-p", "--predict"):
            predict(arg)


def train(trainfile):
    dataset = pd.read_csv(trainfile)
    dataset_new = extract_features(dataset)
    array = dataset_new.values
    n = len(dataset_new.columns)
    X = array[:, 0:n - 1]
    Y = array[:, n - 1]
    seed = 7
    X_train, X_rest, Y_train, Y_rest = model_selection.train_test_split(X, Y, test_size=0.40, random_state=seed)
    X_validation, X_test, Y_validation, Y_test = model_selection.train_test_split(X_rest, Y_rest, test_size=0.50,random_state=seed)
    lm = LinearRegression()
    model = lm
    lm.fit(X_train, Y_train)
    print("done training...")
    model_stats(lm,X_validation,Y_validation)
    pickle.dump(lm, open(saved_model_file, 'wb'))
    print("model saved")
    return lm


def model_stats(lm,X_validation, Y_validation):
    print("score= ", lm.score(X_validation, Y_validation))
    y_predict = lm.predict(X_validation)
    regression_model_mse = mean_squared_error(y_predict, Y_validation)
    print("regression rmse:", math.sqrt(regression_model_mse))


def extract_features(dataset):
    grouped = dataset.groupby('customerID', as_index=False)
    invoice_count = grouped.agg({"invoiceNumber": "count"})
    invoice_count.columns = ['customerID', 'total']

    delayed = dataset[(dataset.DaysLate > 0)]
    delayed = delayed.groupby('customerID', as_index=False)
    delayed_invoice_count = delayed.agg({'invoiceNumber': 'count'})
    delayed_invoice_count.columns = ['customerID', 'delayed']

    delayed_days_avg = delayed.agg({'DaysLate': 'mean'})
    delayed_days_avg.columns = ['customerID', 'avgDaysDelayed']

    settled_days_avg = grouped.agg({'DaysToSettle': 'mean'})
    settled_days_avg.columns = ['customerID', 'avgDaysToSettle']

    invoice_count_stats = pd.merge(invoice_count, delayed_invoice_count, on='customerID', how='left').fillna(0)
    invoice_count_stats = invoice_count_stats.sort_values('customerID')
    invoice_count_stats['paid'] = invoice_count_stats['total'] - invoice_count_stats['delayed']
    invoice_count_stats['delayRatio'] = (invoice_count_stats['delayed'] / invoice_count_stats['total'])

    paid_tot = grouped.agg({"InvoiceAmount": "sum"})
    paid_tot.columns = ['customerID', 'totalAmt']
    delayed_tot = delayed.agg({"InvoiceAmount": "sum"})
    delayed_tot.columns = ['customerID', 'delayedAmt']

    invoice_amt_stats = pd.merge(paid_tot, delayed_tot, on='customerID', how='left').fillna(0)
    invoice_amt_stats['paidAmt'] = invoice_amt_stats['totalAmt'] - invoice_amt_stats['delayedAmt']
    invoice_amt_stats['delayAmtRatio'] = (invoice_amt_stats['delayedAmt'] / invoice_amt_stats['totalAmt'])

    payer_stats = pd.merge(invoice_count_stats, invoice_amt_stats, on="customerID", how='left')
    payer_stats = pd.merge(payer_stats, delayed_days_avg, on="customerID", how="left").fillna(0)
    payer_stats = pd.merge(payer_stats, settled_days_avg, on="customerID", how="left").fillna(0)

    dataset['invoicemonth'] = pd.to_datetime(dataset['InvoiceDate']).dt.month
    dataset['invoicedate'] = pd.to_datetime(dataset['InvoiceDate']).dt.day
    dataset['invoiceday'] = pd.to_datetime(dataset['InvoiceDate']).dt.weekday
    dataset['monthend'] = np.where(dataset['invoicedate'] > 27, 1, 0)
    dataset['firsthalfmonth'] = np.where(dataset['invoicedate'] < 16, 1, 0)
    paperless = {'Paper': 0, 'Electronic': 1}
    dataset['paperless'] = dataset['PaperlessBill'].map(paperless)

    dataset_new = pd.merge(dataset, payer_stats, on='customerID', how='left').fillna(0)

    custlist = payer_stats['customerID'].tolist()
    cust_dict = {x: custlist.index(x) for x in custlist}
    dataset_new['cust'] = dataset_new['customerID'].map(cust_dict)
    dataset_new = dataset_new[['cust', 'InvoiceAmount', 'total', 'totalAmt', 'avgDaysToSettle', 'DaysToSettle']]

    cols = dataset_new.columns
    dataset_new[cols] = dataset_new[cols].apply(pd.to_numeric)

    return dataset_new


def calc_from_features(dataset):

    dataset['invoicemonth'] = pd.to_datetime(dataset['InvoiceDate']).dt.month
    dataset['invoicedate'] = pd.to_datetime(dataset['InvoiceDate']).dt.day
    dataset['invoiceday'] = pd.to_datetime(dataset['InvoiceDate']).dt.weekday
    dataset['monthend'] = np.where(dataset['invoicedate'] > 27, 1, 0)
    dataset['firsthalfmonth'] = np.where(dataset['invoicedate'] < 16, 1, 0)
    paperless = {'Paper': 0, 'Electronic': 1}
    dataset['paperless'] = dataset['PaperlessBill'].map(paperless)
    return dataset

def auto_extract_feature():
    rfe = RFE(model, 4)
    fit = rfe.fit(X_train, Y_train)
    print("Num Features: %d" % fit.n_features_)
    print("Selected Features: %s" % fit.support_)
    print("Feature Ranking: %s" % fit.ranking_)


def file_to_list(filename):
    invoices = pd.read_csv(filename)
    invoices_new = calc_from_features(invoices)

    return []


def predict(datafile):
    x_value = file_to_list(datafile)
    loaded_model = pickle.load(open(saved_model_file, 'rb'))
    y_value=loaded_model.predict(x_value)
    to_json(x_value,y_value)


def to_json():
    print('json')


if __name__ == "__main__":
   main(sys.argv[1:])
