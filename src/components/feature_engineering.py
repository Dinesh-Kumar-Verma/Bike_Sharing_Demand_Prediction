from src.utils.logger import get_logger
import pandas as pd
import os


logger = get_logger(name='feature_engineering', log_file='feature_engineering.log')

def feature_engineering():
    logger.info('\n' + '-' * 80)
    logger.info('Starting feature_engineering...')
    try:
        # load data
        df = pd.read_csv(r'C:\Users\Dinesh Kumar Verma\Desktop\Bike_Sharing_Demand_Prediction\data\processed\bike_sharing_data_processed.csv', sep=',',encoding='latin')
       
        # Encode holiday and fuc_day features
        df['holiday'] = df['holiday'].map({"Holiday": 1, "No Holiday" : 0})
        df['func_day'] = df['func_day'].map({'Yes': 1, 'No': 0})  
        
        #  One hot encoding of hour, month and weekday column
        df_hour = pd.get_dummies(df['hour'], drop_first = True)
        df_hour.columns = df_hour.columns.astype(str)
        df_month = pd.get_dummies(df['month'], drop_first = True)
        df_weekday = pd.get_dummies(df['weekday'], drop_first = True)
        
        #concatinate df_hour, df_month and df_weekday into df
        df = pd.concat([df, df_hour, df_month, df_weekday], axis = 1)
        
        # After one hot encoding, drop hour, month and weekday columns
        df.drop(['hour', 'month', 'weekday'], axis= 1, inplace = True)

        #df.drop(['dew_point_temp'], axis = 1 , inplace = True)
        df.drop(['year'], axis = 1, inplace = True)
        # Simplify column names (optional, but can improve readability)
        df.columns = [col.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace("'", "") for col in df.columns]

        # Save the feature-engineered data
        output_path = r'C:\Users\Dinesh Kumar Verma\Desktop\Bike_Sharing_Demand_Prediction\data\processed\bike_sharing_data_features.csv'
        df.to_csv(output_path, index=False)
        logger.info('Feature-engineered data saved to data/processed/bike_sharing_data_features.csv')
    except Exception as e:
        logger.exception("Failed during feature engineering: %s", e)
    logger.info('Completed feature_engineering.')

if __name__ == '__main__':
    feature_engineering()
