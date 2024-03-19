import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Tabular

# Numerical analysis

def tabular_analysis(df, drop_na_col=False, drop_na_row=False, drop_outliers=False):
    # Describe data
    print('--- Describe ---\n', df.describe(), end='\n\n')

    # Non-numerical columns
    non_numericals = df.select_dtypes(include=['object']).columns.tolist()
    if non_numericals:
        print('Non-numerical columns: {}'.format(non_numericals), end='\n\n')

    # NaN analysis
    nan_columns = df.columns[df.isna().any(axis=0)].tolist()
    if nan_columns:
        print('NaN exists columns: {}'.format(nan_columns))
        print(df.isna().sum())
    
    nan_rows = df.index[df.isna().any(axis=1)].tolist()
    if nan_rows:
        cnt = len(nan_rows)
        print('NaN exists rows: {} ({}%)'.format(cnt, 100 * cnt / len(df)), end='\n\n')

    if drop_na_row:
        df = df.dropna(axis=0)
    if drop_na_col:
        df = df.dropna(axis=1)

    # Outliers analysis
    numericals = df.select_dtypes(exclude=['object']).columns.tolist()
    Q1 = df[numericals].quantile(0.25)
    Q3 = df[numericals].quantile(0.75)
    IQR = Q3 - Q1
    outliers = (df[numericals] < (Q1 - 1.5 * IQR)) | (df[numericals] > (Q3 + 1.5 * IQR))
    print('--- Outliers ---\n', outliers.sum(), end='\n\n')
    if drop_outliers:
        df = df[~outliers.any(axis=1)]
    
    # Correlation analysis
    print('--- Correlation ---\n', df[numericals].corr(), end='\n\n')

# Visualize

def set_style(title, xlabel, ylabel):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

# kwargs: title, xlabel, ylabel, save_path
def scatter(df, x, y, hue=None, **kwargs):
    sns.scatterplot(data=df, x=x, y=y, hue=hue)
    
    set_style(
        kwargs['title'] if 'title' in kwargs else None,
        kwargs['xlabel'] if 'xlabel' in kwargs else x,
        kwargs['ylabel'] if 'ylabel' in kwargs else y,
    )
    if 'save_path' in kwargs:
        plt.savefig(kwargs['save_path'])
    plt.show()

# kwargs: title, xlabel, ylabel, save_path
def regression(df, x, y, hue=None, col=None, **kwargs):
    sns.lmplot(data=df, x=x, y=y, hue=hue, col=col)
    
    set_style(
        kwargs['title'] if 'title' in kwargs else None,
        kwargs['xlabel'] if 'xlabel' in kwargs else x,
        kwargs['ylabel'] if 'ylabel' in kwargs else y,
    )
    if 'save_path' in kwargs:
        plt.savefig(kwargs['save_path'])
    plt.show()


# kind: strip(default), swarm, box, violin, boxen, point, bar, count
# kwargs: title, xlabel, ylabel, save_path
def categorize_plot(df, x, y, kind='strip', **kwargs):
    sns.catplot(data=df, x=x, y=y, kind=kind)
    
    set_style(
        kwargs['title'] if 'title' in kwargs else None,
        kwargs['xlabel'] if 'xlabel' in kwargs else None,
        kwargs['ylabel'] if 'ylabel' in kwargs else None,
    )
    if 'save_path' in kwargs:
        plt.savefig(kwargs['save_path'])
    plt.show()


# Images

def img_show(img):
    import torch
    if type(img) == torch.Tensor:
        plt.imshow(img.permute(1, 2, 0))
    else:
        plt.imshow(img)
    plt.axis('off')
    plt.show()

def imageset_metadata(datasets):
    print('ImageSet shape:', datasets.shape)
    print('Image shape:', datasets[0].shape)

    channels = len(datasets[0].shape)
    metadata = [[], [], []]
    for idx in range(channels):
        metadata[idx] = [
            int(datasets[:, idx, :, :].max() * 255),
            int(datasets[:, idx, :, :].min() * 255),
            float(datasets[:, idx, :, :].mean() * 255),
            float(datasets[:, idx, :, :].std() * 255)
        ]
    df = pd.DataFrame({
        'Red': metadata[0],
        'Green': metadata[1],
        'Blue': metadata[2]
    }, index=['Max', 'Min', 'Mean', 'Std'])
    
    print(df)