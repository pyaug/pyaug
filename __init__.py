import pandas as pd
import random
import numpy as np



def PyAugLinear(original_dataframe, augmented_rows, percent, ignore_cols):

    '''PyAugLinear ingests a Pandas DataFrame and creates a new Pandas DataFrame consisting of augmented_rows number of
    rows. Columns selected for augmentation are perturbed using a linear distribution with a maximum perturbation
    defined using the percent argument.'''

    df_rows = original_dataframe.shape[0] #Calculate number of rows in the DataFrame.
    df_columns = original_dataframe.shape[1] #Calculate number of columns in the DataFrame.

    number_dataframes = list(range(0, int(augmented_rows / df_rows) + 1)) #Additional DataFrame to get enough augmented rows.

    try:
        if augmented_rows < df_rows:
            message = 'Error: Number of Augmented Rows < Number of Input Rows' #Produce error message.
            return message

        else:
            #Evaluate correct number of decimal places.
            random_row = original_dataframe.sample(1).values.tolist() #Select random row.
            decimal_number = (len(str(random_row[0][1]).replace('.', '')) - 1) #Count number of decimal places.


            if all(type(item) is str for item in ignore_cols) == True: #Check if all items within ignore cols is string type.

                df_perturbation = {index: pd.DataFrame(pd.DataFrame(np.random.uniform((100 - percent), (100 + percent), size=(df_rows, df_columns)), columns=original_dataframe.columns.values)/100) for index in number_dataframes}
                df_perturbation = pd.concat(df_perturbation.values(), ignore_index=True)

                df_perturbation[ignore_cols] = 1 #Set ignored columns to 1.

                df_realdata = {index: original_dataframe for index in number_dataframes}
                df_realdata = pd.concat(df_realdata.values(), ignore_index=True)

                perturbed_realdata = df_realdata.mul(df_perturbation).round(decimal_number).head(augmented_rows) #Multiply DataFrames and round.


            elif all(type(item) is int for item in ignore_cols) == True: #Check if all items within ignore cols is int type.

                columns = (original_dataframe.columns.tolist())
                ignore_cols = [columns[i - 1] for i in ignore_cols] #Convert DataFrame[int] to DataFrame[string]

                df_perturbation = {index: pd.DataFrame(pd.DataFrame(np.random.randint((100 - percent), (100 + percent), size=(df_rows, df_columns)), columns=original_dataframe.columns.values)/100) for index in number_dataframes}
                df_perturbation = pd.concat(df_perturbation.values(), ignore_index=True)

                df_perturbation[ignore_cols] = 1 #Set ignored columns to 1.

                df_realdata = {index: original_dataframe for index in number_dataframes}
                df_realdata = pd.concat(df_realdata.values(), ignore_index=True)

                perturbed_realdata = df_realdata.mul(df_perturbation).round(decimal_number).head(augmented_rows) #Multiply DataFrames and round.


            else:
                message = 'Error: Enter Valid ignore_cols Format' #Produce error message.
                return message


    except IndexError:
        message = 'IndexError' #Produce error message.
        return message

    return perturbed_realdata


def PyAugNormal(original_dataframe, augmented_rows, std, ignore_cols):

    '''PyAugNormal ingests a Pandas DataFrame and creates a new Pandas DataFrame consisting of augmented_rows number of
    rows. Columns selected for augmentation are perturbed using a normal distribution with a maximum perturbation
    standard deviation using the std argument.'''

    df_rows = original_dataframe.shape[0] #Calculate number of rows in DataFrame.
    df_columns = original_dataframe.shape[1] #Calculate number of columns in DataFrame.

    number_dataframes = list(range(0, int(augmented_rows / df_rows) + 1)) #Additional DataFrame to get enough augmented rows.

    try:
        if augmented_rows < df_rows:
            message = 'Error: Number of Augmented Rows < Number of Input Rows' #Produce error message.
            return message

        else:
            #Evaluating Correct number of decimal places.
            random_row = original_dataframe.sample(1).values.tolist() #Select random row.
            decimal_number = (len(str(random_row[0][1]).replace('.', '')) - 1) #Count number of decimal places.


            if all(type(item) is str for item in ignore_cols) == True: #Check if all items within ignore cols is string type.

                df_perturbation = {index: pd.DataFrame(pd.DataFrame(np.random.normal(100, std, size=(df_rows, df_columns)), columns=original_dataframe.columns.values)/100) for index in number_dataframes}
                df_perturbation = pd.concat(df_perturbation.values(), ignore_index=True)

                df_perturbation[ignore_cols] = 1 #Set ignored columns to 1.

                df_realdata = {index: original_dataframe for index in number_dataframes}
                df_realdata = pd.concat(df_realdata.values(), ignore_index=True)

                perturbed_realdata = df_realdata.mul(df_perturbation).round(decimal_number).head(augmented_rows) #Multiply DataFrames and round.


            elif all(type(item) is int for item in ignore_cols) == True: #Check if all items within ignore cols is int type.

                columns = (original_dataframe.columns.tolist())
                ignore_cols = [columns[i - 1] for i in ignore_cols] #Convert DataFrame[int] to DataFrame[string]

                df_perturbation = {index: pd.DataFrame(pd.DataFrame(np.random.normal(100, std, size=(df_rows, df_columns)), columns=original_dataframe.columns.values)/100) for index in number_dataframes}
                df_perturbation = pd.concat(df_perturbation.values(), ignore_index=True)

                df_perturbation[ignore_cols] = 1 #Set ignored columns to 1.

                df_realdata = {index: original_dataframe for index in number_dataframes}
                df_realdata = pd.concat(df_realdata.values(), ignore_index=True)

                perturbed_realdata = df_realdata.mul(df_perturbation).round(decimal_number).head(augmented_rows) #Multiply DataFrames and round.


            else:
                message = 'Error: Enter Valid ignore_cols Format'  #Produce error message.
                return message


    except IndexError:
        message = 'IndexError' #Produce error message.
        return message

    return perturbed_realdata


def PyAugLogistic(original_dataframe, augmented_rows, std, ignore_cols):

    '''PyAugLogistic ingests a Pandas DataFrame and creates a new Pandas DataFrame consisting of augmented_rows number of
    rows. Columns selected for augmentation are perturbed using a logistic distribution with a maximum perturbation
    standard deviation using the std argument.'''

    df_rows = original_dataframe.shape[0] #Calculate number of rows in DataFrame.
    df_columns = original_dataframe.shape[1] #Calculate number of columns in DataFrame.

    number_dataframes = list(range(0, int(augmented_rows / df_rows) + 1)) #Additional DataFrame to get enough augmented rows.

    try:
        if augmented_rows < df_rows:
            message = 'Error: Number of Augmented Rows < Number of Input Rows' #Produce error message.
            return message

        else:
            #Evaluating Correct number of decimal places.
            random_row = original_dataframe.sample(1).values.tolist() #Select random row.
            decimal_number = (len(str(random_row[0][1]).replace('.', '')) - 1) #Count number of decimal places.


            if all(type(item) is str for item in ignore_cols) == True: #Check if all items within ignore cols is string type.

                df_perturbation = {index: pd.DataFrame(pd.DataFrame(np.random.logistic(100, std, size=(df_rows, df_columns)), columns=original_dataframe.columns.values)/100) for index in number_dataframes}
                df_perturbation = pd.concat(df_perturbation.values(), ignore_index=True)

                df_perturbation[ignore_cols] = 1 #Set ignored columns to 1.

                df_realdata = {index: original_dataframe for index in number_dataframes}
                df_realdata = pd.concat(df_realdata.values(), ignore_index=True)

                perturbed_realdata = df_realdata.mul(df_perturbation).round(decimal_number).head(augmented_rows) #Multiply DataFrames and round.


            elif all(type(item) is int for item in ignore_cols) == True: #Check if all items within ignore cols is int type.

                columns = (original_dataframe.columns.tolist())
                ignore_cols = [columns[i - 1] for i in ignore_cols] #Convert DataFrame[int] to DataFrame[string]

                df_perturbation = {index: pd.DataFrame(pd.DataFrame(np.random.logistic(100, std, size=(df_rows, df_columns)), columns=original_dataframe.columns.values)/100) for index in number_dataframes}
                df_perturbation = pd.concat(df_perturbation.values(), ignore_index=True)

                df_perturbation[ignore_cols] = 1 #Set ignored columns to 1.

                df_realdata = {index: original_dataframe for index in number_dataframes}
                df_realdata = pd.concat(df_realdata.values(), ignore_index=True)

                perturbed_realdata = df_realdata.mul(df_perturbation).round(decimal_number).head(augmented_rows) #Multiply DataFrames and round.


            else:
                message = 'Error: Enter Valid ignore_cols Format' #Produce error message.
                return message


    except IndexError:
        message = 'IndexError' #Produce error message.
        return message

    return perturbed_realdata


def PyAugLaplace(original_dataframe, augmented_rows, std, ignore_cols):

    '''PyAugLaplace ingests a Pandas DataFrame and creates a new Pandas DataFrame consisting of augmented_rows number of
    rows. Columns selected for augmentation are perturbed using a Laplace distribution with a maximum perturbation
    standard deviation using the std argument.'''

    df_rows = original_dataframe.shape[0] #Calculate number of rows in DataFrame.
    df_columns = original_dataframe.shape[1] #Calculate number of columns in DataFrame.

    number_dataframes = list(range(0, int(augmented_rows / df_rows) + 1)) #Additional DataFrame to get enough augmented rows.

    try:
        if augmented_rows < df_rows:
            message = 'Error: Number of Augmented Rows < Number of Input Rows' #Produce error message.
            return message

        else:
            #Evaluating Correct number of decimal places.
            random_row = original_dataframe.sample(1).values.tolist() #Select random row.
            decimal_number = (len(str(random_row[0][1]).replace('.', '')) - 1) #Count number of decimal places.


            if all(type(item) is str for item in ignore_cols) == True: #Check if all items within ignore cols is string type.

                df_perturbation = {index: pd.DataFrame(pd.DataFrame(np.random.laplace(100, std, size=(df_rows, df_columns)), columns=original_dataframe.columns.values)/100) for index in number_dataframes}
                df_perturbation = pd.concat(df_perturbation.values(), ignore_index=True)

                df_perturbation[ignore_cols] = 1 #Set ignored columns to 1.

                df_realdata = {index: original_dataframe for index in number_dataframes}
                df_realdata = pd.concat(df_realdata.values(), ignore_index=True)

                perturbed_realdata = df_realdata.mul(df_perturbation).round(decimal_number).head(augmented_rows) #Multiply DataFrames and round.


            elif all(type(item) is int for item in ignore_cols) == True: #Check if all items within ignore cols is int type.

                columns = (original_dataframe.columns.tolist())
                ignore_cols = [columns[i - 1] for i in ignore_cols] #Convert DataFrame[int] to DataFrame[string]

                df_perturbation = {index: pd.DataFrame(pd.DataFrame(np.random.laplace(100, std, size=(df_rows, df_columns)), columns=original_dataframe.columns.values)/100) for index in number_dataframes}
                df_perturbation = pd.concat(df_perturbation.values(), ignore_index=True)

                df_perturbation[ignore_cols] = 1 #Set ignored columns to 1.

                df_realdata = {index: original_dataframe for index in number_dataframes}
                df_realdata = pd.concat(df_realdata.values(), ignore_index=True)

                perturbed_realdata = df_realdata.mul(df_perturbation).round(decimal_number).head(augmented_rows) #Multiply DataFrames and round.


            else:
                message = 'Error: Enter Valid ignore_cols Format' #Produce error message.
                return message


    except IndexError:
        message = 'IndexError' #Produce error message.
        return message

    return perturbed_realdata