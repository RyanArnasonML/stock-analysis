from distutils.core import setup

setup(
    name='stock_analysis',
    version='0.1',
    description='Classes for technical analysis of stocks.',
    author='Stefanie Molin',
    author_email='24376333+stefmolin@users.noreply.github.com',
    license='MIT',
    url='https://github.com/stefmolin/stock-analysis',
    packages=['stock_analysis'],
    install_requires=[
        'matplotlib>=3.0.2',
        'numpy>=1.15.2',
        'pandas>=0.23.4',
        'pandas-datareader==0.9.0',
        'seaborn>=0.9.0',
        'pykalman>=0.9.5',
        'mplfinance>=0.12.7a4',
        'statsmodels>=0.12.1'
    ],
)