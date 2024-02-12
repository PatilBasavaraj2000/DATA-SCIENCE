.. code:: ipython3

    # Import necessary libraries
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    
    # Generate synthetic sales data for demonstration purposes
    np.random.seed(42)
    advertising_expenditure = np.random.rand(100) * 50
    sales = 10 + 2 * advertising_expenditure + np.random.randn(100) * 5
    
    # Create a DataFrame with the synthetic data
    sales_data = pd.DataFrame({'Advertising_Expenditure': advertising_expenditure, 'Sales': sales})
    
    # Visualize the relationship with a scatter plot
    plt.scatter(sales_data['Advertising_Expenditure'], sales_data['Sales'])
    plt.xlabel('Advertising Expenditure')
    plt.ylabel('Sales')
    plt.title('Scatter Plot: Advertising Expenditure vs Sales')
    plt.show()
    
    # Separate features (X) and target variable (y)
    X = sales_data[['Advertising_Expenditure']]
    y = sales_data['Sales']
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build a Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared Score: {r2}")
    



.. image:: output_0_0.png


.. parsed-literal::

    Mean Squared Error: 16.34248784292508
    R-squared Score: 0.9825431689004598
    

