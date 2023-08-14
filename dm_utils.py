import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors


# fixed tf seed
import tensorflow as tf
tf.random.set_seed(42)

class DBMbuilder:
    def __init__(self, clf, ssnp, scaler2d, scalernd):
        self.clf = clf
        self.ssnp = ssnp
        self.scaler2d = scaler2d
        self.scalernd = scalernd
        self.xx, self.yy = self.make_meshgrid()
        self.map_res = self.get_prob_map()
        self.inversed_feature_res = self.inversed_feature()
        
    
    def make_meshgrid(self, x=np.array([0,1]), y=np.array([0,1]), grid=300):
        """Create a mesh of points to plot in

        Parameters
        ----------
        x: data to base x-axis meshgrid on
        y: data to base y-axis meshgrid on
        h: stepsize for meshgrid, optional

        Returns
        -------
        xx, yy : ndarray
        """
        x_min, x_max = x.min() - 0.0, x.max() + 0.0
        y_min, y_max = y.min() - 0.0, y.max() + 0.0
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid),
                            np.linspace(y_min, y_max, grid))
        return xx, yy
    
    def get_prob_map(self):
        """Get probability map for the classifier
        """
        xx, yy = self.xx, self.yy
        inverse_scaler = self.scaler2d.inverse_transform(np.c_[xx.ravel(), yy.ravel()]).astype('float32')
        inversed = self.ssnp.inverse_transform(inverse_scaler).astype('float32')
        probs = self.clf.predict_proba(inversed)
        alpha = np.amax(probs, axis=1)
        labels = probs.argmax(axis=1)
        alpha = alpha.reshape(xx.shape)
        labels = labels.reshape(xx.shape)
        # self.map_res = (alpha, labels, xx, yy)
        return alpha, labels
    

    def plot_prob_map(self, ax=None, cmap=cm.Set1, ture_map=False):
        """Plot probability map for the classifier
        """
        if not self.map_res:
            alpha, labels= self.get_prob_map()
        else:
            alpha, labels= self.map_res
        if ax is None:
            ax = plt.gca()
        xx, yy = self.xx, self.yy
        map = cmap(labels/self.clf.classes_.max())
        map[:, :, 3] = alpha    #np.max(alpha - 0.1, 0, axis=0 ,keepdims=True)
        map = np.flip(map, 0)
        ax.imshow(map, interpolation='nearest', aspect='auto', extent=[xx.min(), xx.max(), yy.min(), yy.max()])
        ax.set_facecolor('black')
        ax.set_xticks([])
        ax.set_yticks([])
        # set lim
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        return ax


        
    def inversed_feature(self):
        xx, yy = self.xx, self.yy
        scaled_2d = self.scaler2d.inverse_transform(np.c_[xx.ravel(), yy.ravel()]).astype('float32')
        inversed = self.ssnp.inverse_transform(scaled_2d).astype('float32')
        labels = self.clf.predict(inversed)
        inversed = self.scalernd.inverse_transform(inversed).astype('float32')
        # labels_ssnp = self.ssnp.predict2d(scaled_2d)

        # num_features = inversed.shape[1]
        res = inversed.reshape(xx.shape[0], xx.shape[1], -1)
        labels = labels.reshape(xx.shape[0], xx.shape[1])
        # labels_ssnp = labels_ssnp.reshape(xx.shape[0], xx.shape[1])
        return res #, labels #labels_ssnp

    def plot_inversed_feature(self, ind, feature_names, ax=None, countour=True, **params): ##
        if not ax:
            ax = plt.gca()
        fig = ax.get_figure()

        res = self.inversed_feature_res
        labels = self.map_res[1]

        xx, yy = self.xx, self.yy
        
        feature_plot = 10 ** res[:, :, ind] -1
        # feature_plot = np.flipud(feature_plot)
        if countour:
            temmap = ax.contourf(xx, yy, feature_plot, cmap='bwr', norm=colors.LogNorm(vmin=feature_plot.min(), vmax=feature_plot.max()),  **params) #
            
        else:
            temmap = ax.imshow(np.flip(feature_plot, 0), cmap='bwr', norm=colors.LogNorm(vmin=feature_plot.min(), vmax=feature_plot.max()), extent=[xx.min(), xx.max(), yy.min(), yy.max()])
            # ax.invert_yaxis()
        cnt = ax.contour(xx, yy, labels, **params, colors='k')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('{}'.format(feature_names[ind]))
        # invert y axis
        # color bar
        cbar = fig.colorbar(temmap, ax=ax)
        return ax

    def transform(self, X, normed=True):
        if not normed:
            X = np.log10(X+1)
            X = self.scalernd.transform(X)
        X2d = self.ssnp.transform(X)
        X2d = self.scaler2d.transform(X2d)
        return X2d


##############################################
class DBMsearch:
    def __init__(self, models, ssnp, X, y, cv):
        self.models = models
        self.ssnp = ssnp
        self.X = X
        self.y = y
        self.cv = cv
        self.best_model = None
        self.results = pd.DataFrame(columns=['model','classifier accucarcy',  "DBM accucarcy", 'consistancy'])

    def one_search(self, model):
        clf = model
        clf_acc = []
        dbm_acc = []
        projection_miss = []
        for train_index, test_index in self.cv.split(self.X, self.y):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            clf.fit(X_train, y_train)
            # clf.predict(X_test)
            clf_acc.append(self.clf_missclass(clf, X_test, y_test))
            dbm_acc.append(self.dbm_missclass(clf, X_test, y_test))
            projection_miss.append(self.projection_miss(clf, X_test, y_test))
        clf_acc = np.array(clf_acc)
        dbm_acc = np.array(dbm_acc)
        projection_miss = np.array(projection_miss)
        ### update self.results
        self.results = self.results.append({'model': model, 'classifier accucarcy': clf_acc.mean(),  "DBM accucarcy": dbm_acc.mean(), 'consistancy': projection_miss.mean()}, ignore_index=True)
        # self.results = self.results.append({'model': model, 'clf acc. mean': clf_acc.mean(), 'clf acc. std': clf_acc.std(), "DBM acc. mean": dbm_acc.mean(), 'DBM acc. std': dbm_acc.std(), 'projection miss mean': projection_miss.mean(), 'projection miss std': projection_miss.std()}, ignore_index=True)
        
        return clf_acc.mean(), dbm_acc.mean(), projection_miss.mean()

    def search(self):
        for model in self.models:
            print('searching model: {}'.format(model))
            self.one_search(model)
        # # to numeric
        # self.results['classifier accucarcy'] = pd.to_numeric(self.results['classifier accucarcy'])
        # self.results['DBM accucarcy'] = pd.to_numeric(self.results['DBM accucarcy'])
        # self.results['consistancy'] = pd.to_numeric(self.results['consistancy'])
        # # find best model
        print(self.results)
        idx = self.results['DBM accucarcy'].idxmax()
        self.best_model = self.results.loc[idx, 'model']
        print('best model: {}'.format(self.best_model))
        return self.best_model

        
    def projection_miss(self,clf, x, y=None):
    # project x to the latent space and then back to the original space
        # time0 = time.time()
        x2d = self.ssnp.transform(x)
        if clf == None:
            pred2d = self.ssnp.predict2d(x2d)
            y_pred = self.ssnp.predict2d(x2d) #??
        else:
            xnd = self.ssnp.inverse_transform(x2d)
            pred2d = clf.predict(xnd)
            y_pred = clf.predict(x)
        # return the index where pred2d != y_pred
        ind = np.where(pred2d == y_pred)[0]
        return len(ind)/len(y_pred)

    def clf_missclass(self, clf, x, y):
        # time0 = time.time()
        y_pred = clf.predict(x)
        # print('time: clf_missclass:', time.time() - time0)
        ind = np.where(y_pred == y)[0]
        return len(ind) / len(y)

    def dbm_missclass(self, clf, x, y):
        # time0 = time.time()
        # project x to the latent space and then back to the original space
        x2d = self.ssnp.transform(x)
        if clf == None:
            pred2d = self.ssnp.predict2d(x2d)
        
        else:
            xnd = self.ssnp.inverse_transform(x2d)
            pred2d = clf.predict(xnd)
        return len(np.where(pred2d == y)[0]) / len(y)

        