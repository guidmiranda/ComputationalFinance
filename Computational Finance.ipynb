{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "f1ea1349",
   "metadata": {},
   "outputs": [],
   "source": [
    "#---------------------------------------------------------------------------------#\n",
    "#     Computational Finance\n",
    "#     NOVA IMS\n",
    "#     Group Project: Carlos Cardoso | 20211220       Carlota Reis | 20211208   \n",
    "#                    Guilherme Miranda | 20210420    Mariana Garcia | 20210838 \n",
    "#---------------------------------------------------------------------------------#"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "7a3a11b1",
   "metadata": {},
   "outputs": [],
   "source": [
    "import yfinance as yf\n",
    "import datetime\n",
    "import time\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "from scipy.optimize import minimize\n",
    "from scipy import optimize\n",
    "import glob"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "396c36ac",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Exercise 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "37652c22",
   "metadata": {},
   "outputs": [],
   "source": [
    "#a)\n",
    "\n",
    "#Building a class that receives, as inputs, maturity, an array with prices and an array with coupons\n",
    "class YieldCurve:\n",
    "    def __init__(self,maturity,price,coupon):\n",
    "        self.maturity = maturity\n",
    "        self.price = price\n",
    "        self.coupon = coupon"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "e347feaa",
   "metadata": {},
   "outputs": [],
   "source": [
    "#b)\n",
    "\n",
    "class YieldCurve:\n",
    "    def __init__(self,maturity,price,coupon):\n",
    "        self.maturity = maturity\n",
    "        self.price = price\n",
    "        self.coupon = coupon\n",
    "        \n",
    "#Adding a method to the YieldCurve class that, given the inputs of the class, returns the discount factors \n",
    "\n",
    "    def matrix_operation(self):\n",
    "        years = self.maturity\n",
    "        \n",
    "        #creating an array with cashflows for each bond\n",
    "        cfs = [[coupon if year < maturity else \\\n",
    "        coupon + 100 if year == maturity else 0 \\\n",
    "        for year in range(years)] \\\n",
    "        for maturity, coupon in enumerate(self.coupon)]\n",
    "        \n",
    "        #Creating a vector of discount factors using matrices operations\n",
    "        dfs = np.linalg.inv(cfs) @ self.price\n",
    "        return dfs\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "ec9cdab7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0.95172414, 0.90461408, 0.86124827, 0.8227426 , 0.78916271,\n",
       "       0.76042723, 0.73611886, 0.71559687, 0.69849544, 0.68423626])"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Providing the initial inputs for the YieldCurve class\n",
    "coupon = np.linspace(1.50,3.75,10)\n",
    "price = [96.6, 93.71, 91.56, 90.24, 89.74, 90.04, 91.09, 92.82, 95.19, 98.14]\n",
    "maturity = 10\n",
    "\n",
    "#Using the matrix_operation method to get the discount factors \n",
    "discount_factors=YieldCurve(maturity,price,coupon)\n",
    "(discount_factors.matrix_operation())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "74281876",
   "metadata": {},
   "outputs": [],
   "source": [
    "#c)\n",
    "\n",
    "class YieldCurve:\n",
    "    def __init__(self,maturity,price,coupon):\n",
    "        self.maturity = maturity\n",
    "        self.price = price\n",
    "        self.coupon = coupon\n",
    "        \n",
    "\n",
    "    def matrix_operation(self):\n",
    "        years = self.maturity\n",
    "        cfs = [[coupon if year < maturity else \\\n",
    "        coupon + 100 if year == maturity else 0 \\\n",
    "        for year in range(years)] \\\n",
    "        for maturity, coupon in enumerate(self.coupon)]\n",
    "        dfs = np.linalg.inv(cfs) @ self.price\n",
    "        return dfs\n",
    "    \n",
    "#creating the global solver method \n",
    "\n",
    "    def global_solver(self):\n",
    "        \n",
    "        #Defining a function that, using the inputs from the Yieldcurve class and for a set of discount factors, returns the price of bonds\n",
    "        #It is the objective function of the minimization problem\n",
    "        def bond_price(disc_factors):\n",
    "            years = self.maturity\n",
    "            cfs = [[coupon if year < maturity else coupon + 100 if year == maturity else 0 for year in range(years)] for maturity, coupon in enumerate(self.coupon)]\n",
    "            cfs = pd.DataFrame(cfs)\n",
    "            bond_price = cfs@disc_factors\n",
    "            return bond_price\n",
    " \n",
    "\n",
    "        #Defining an error function that returns the sum of the squared difference between the bond prices calculated with the bond_price function and a set of bond prices given as input \n",
    "        def error(disc_factors): \n",
    "            return ((np.array(bond_price(disc_factors)) - np.array(price))**2).sum()\n",
    "            \n",
    "\n",
    "        #Defining an initial set of values for the discount factors - initial values for the optimization process\n",
    "        disc_factors0 = np.array([0.85, 0.45, 0.67, 0.047, 0.49, 0.43,0.057,0.07,0.38,0.02])\n",
    "        \n",
    "        #Minimizing the error function\n",
    "        #By minimizing this difference, we arrive to a set of discount factors\n",
    "        disc_factors = minimize(error,disc_factors0)['x']\n",
    "        return(disc_factors)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "7e3a066c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0.95172413, 0.90461408, 0.86124826, 0.82274259, 0.7891627 ,\n",
       "       0.76042722, 0.73611886, 0.71559687, 0.69849544, 0.68423625])"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Providing the initial inputs for the YieldCurve class\n",
    "coupon = np.array(np.linspace(1.50,3.75,10))\n",
    "price = np.array([96.6, 93.71, 91.56, 90.24, 89.74, 90.04, 91.09, 92.82, 95.19, 98.14])\n",
    "maturity = 10\n",
    "\n",
    "#Using the global_solver method to get the discount factors \n",
    "discount_factors=YieldCurve(maturity,price,coupon)\n",
    "(discount_factors.global_solver())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "46d7d877",
   "metadata": {},
   "outputs": [],
   "source": [
    "#d)\n",
    "\n",
    "class YieldCurve:\n",
    "    def __init__(self,maturity,price,coupon):\n",
    "        self.maturity = maturity\n",
    "        self.price = price\n",
    "        self.coupon = coupon\n",
    "        \n",
    "        \n",
    "    def matrix_operation(self):\n",
    "        years = self.maturity\n",
    "        cfs = [[coupon if year < maturity else \\\n",
    "        coupon + 100 if year == maturity else 0 \\\n",
    "        for year in range(years)] \\\n",
    "        for maturity, coupon in enumerate(self.coupon)]\n",
    "        dfs = np.linalg.inv(cfs) @ self.price\n",
    "        return dfs\n",
    "    \n",
    "    def global_solver(self):\n",
    "        def bond_price(disc_factors):\n",
    "            years = self.maturity\n",
    "            cfs = [[coupon if year < maturity else coupon + 100 if year == maturity else 0 for year in range(years)] for maturity, coupon in enumerate(self.coupon)]\n",
    "            cfs = pd.DataFrame(cfs)\n",
    "            bond_price = cfs@disc_factors\n",
    "            return bond_price\n",
    "        def error(disc_factors): \n",
    "            return ((np.array(bond_price(disc_factors)) - np.array(price))**2).sum()\n",
    "        disc_factors0 = np.array([0.85, 0.45, 0.67, 0.047, 0.49, 0.43,0.057,0.07,0.38,0.02])\n",
    "        disc_factors = minimize(error,disc_factors0)['x']\n",
    "        return(disc_factors)\n",
    "    \n",
    "    \n",
    "    \n",
    "    #creating the iterative procedure method\n",
    "    def it_procedure(self):\n",
    "        \n",
    "        #Creating arrays of the inputs of the class\n",
    "        coupon = np.array(self.coupon)\n",
    "        price = np.array(self.price)\n",
    "        maturity = np.array(range(1,self.maturity+1))\n",
    "        \n",
    "        #creating an array with information, for each bond, of its maturity, price and coupon\n",
    "        bonds = np.array([maturity,price,coupon]).T\n",
    "        maturities, prices, coupons = bonds[:, 0], bonds[:, 1], bonds[:, 2]\n",
    "        years = len(coupons)\n",
    "        \n",
    "        #Starting by creating a matrix of zeros\n",
    "        from scipy import optimize\n",
    "        zeros = np.zeros((len(bonds)))\n",
    "        \n",
    "        #Looping through each bond to create its cashflows\n",
    "        #Using the prices given as input for the Yieldcurve class, go through each bond and calculate the interest rate that gives the price\n",
    "        #Doing it from short to longer maturities and using the previous interest rate to calculate the following\n",
    "        \n",
    "        for bond in bonds:\n",
    "            maturity, price, coupon = bond\n",
    "            coupon /= 100\n",
    "            maturity = int(maturity)\n",
    "            known_cf = sum([coupon/(1+zeros[n])**(n+1) for n in range(maturity -1)]) * 100\n",
    "            f = lambda z : known_cf + ((1 + coupon) * 100)/(1+z)**maturity - price\n",
    "            zero = optimize.newton(f, 0)\n",
    "            zeros[int(maturity)-1] = zero\n",
    "        \n",
    "        #Calculating the discount factors from the obtained interest rates\n",
    "        dfs = np.array([1/(1+z)**(n+1) for n,z in enumerate(zeros)])\n",
    "        return dfs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "0b5ddc44",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0.95172414, 0.90461408, 0.86124827, 0.8227426 , 0.78916271,\n",
       "       0.76042723, 0.73611886, 0.71559687, 0.69849544, 0.68423626])"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Providing the initial inputs for the YieldCurve class\n",
    "coupon = np.linspace(1.50,3.75,10)\n",
    "price = [96.6, 93.71, 91.56, 90.24, 89.74, 90.04, 91.09, 92.82, 95.19, 98.14]\n",
    "maturity = 10\n",
    "\n",
    "#Using the interative procedure method to get the discount factors\n",
    "discount_factors=YieldCurve(maturity,price,coupon)\n",
    "(discount_factors.it_procedure())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "491eb135",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   Spot Rates  Maturity\n",
      "0    0.050725         1\n",
      "1    0.051401         2\n",
      "2    0.051051         3\n",
      "3    0.049987         4\n",
      "4    0.048496         5\n",
      "5    0.046704         6\n",
      "6    0.044738         7\n",
      "7    0.042717         8\n",
      "8    0.040675         9\n",
      "9    0.038674        10\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x7fa36027aa60>]"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#e)\n",
    "\n",
    "#Creating a variable to store the array of discount factors calculated with the matrix_operation method from the YieldCurve class\n",
    "discount_factors=YieldCurve(maturity,price,coupon)\n",
    "d_factors = (discount_factors.matrix_operation())\n",
    "\n",
    "\n",
    "#From the obtained discount factors,loop through each one of them and calculate the corresponding spot rate\n",
    "spot_rates=[]\n",
    "for n in range(1,len(d_factors)+1):\n",
    "    r0 = ((1/d_factors[n-1])**(1/n))-1\n",
    "    \n",
    "    #store the results obtained in each loop\n",
    "    spot_rates.append(r0)\n",
    "\n",
    "#Creating a DataFrame for the obtained spot rates\n",
    "spot_rates = pd.DataFrame(spot_rates)\n",
    "spot_rates.columns=['Spot Rates']\n",
    "spot_rates['Maturity'] = range(1,11)\n",
    "\n",
    "#Visualizing the DataFrame\n",
    "print(spot_rates)\n",
    "\n",
    "#Plotting the obtained spot rates,for each maturity\n",
    "plt.plot(spot_rates['Maturity'],spot_rates['Spot Rates'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "241465a9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0.05072463768115943, 0.0513947974577041, 0.05105360885668887, 0.05002601840806586, 0.048606822953342795, 0.04693075263753644, 0.0451274540652176, 0.043310562386741226, 0.041513813840999184, 0.039790824253531844]\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x7fa352130670>]"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#f)\n",
    "\n",
    "coupon = np.linspace(1.50,3.75,10)\n",
    "price = [96.6, 93.71, 91.56, 90.24, 89.74, 90.04, 91.09, 92.82, 95.19, 98.14]\n",
    "maturity = range(1,len(price)+1)\n",
    "\n",
    "#Creating an array with the characteristics of each bond (maturity, price and coupon)\n",
    "bonds = np.array([maturity,price,coupon]).T\n",
    "\n",
    "\n",
    "#Defining a function that, given the price and the cashflow of a bond, calculates the implicit yield to maturity\n",
    "def bond_ytm(maturity, price, coupon):\n",
    "    ytm_func = lambda y : sum([(coupon if n < maturity else coupon + 1)/(1+y)**n\n",
    "                               for n in range(1, maturity +1)]) * 100 - price\n",
    "    return optimize.newton(ytm_func, 0)\n",
    "\n",
    "#Looping through the information on the bonds array and calculating the implicit yield to maturity of each\n",
    "yields = []\n",
    "for bond in bonds:\n",
    "    maturity, price, coupon = bond\n",
    "    \n",
    "    #storing the result of each loop \n",
    "    yields.append(bond_ytm(int(maturity), price, coupon/100))\n",
    "\n",
    "#Visualizing the obtained yields  \n",
    "print(yields)\n",
    "\n",
    "#Plotting the obtained yields\n",
    "plt.plot(range(1,11),yields)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "71372a73",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "       zero       fwd\n",
      "0  0.050725       NaN\n",
      "1  0.051401  0.051401\n",
      "2  0.051051  0.050702\n",
      "3  0.049987  0.047859\n",
      "4  0.048496  0.044022\n",
      "5  0.046704  0.039535\n",
      "6  0.044738  0.034911\n",
      "7  0.042717  0.030590\n",
      "8  0.040675  0.026382\n",
      "9  0.038674  0.022668\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x7fa35221d3a0>]"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#g)\n",
    "\n",
    "#Creating a DataFrame with the Spot Rates\n",
    "rates = pd.DataFrame({\n",
    "        'zero': spot_rates['Spot Rates']\n",
    "    }, index= list(range(0,10)))\n",
    "\n",
    "#Defining a function that calculates the forward rate using the spot rates (assuming no-arbitrage condition)\n",
    "def makeFwd(df):\n",
    "    df['N'] = df.index\n",
    "    b = df.zero * df.N\n",
    "    df['fwd'] = (b - b.shift()) / (df.N - df.N.shift())\n",
    "    return df[['zero', 'fwd']]\n",
    "\n",
    "\n",
    "#Store the obtained forward rates on the DataFrame\n",
    "rates = makeFwd(rates)\n",
    "\n",
    "#Visualizing the obtained forward rates \n",
    "print(rates)\n",
    "\n",
    "#Plotting the obtained forward\n",
    "plt.plot(rates.index, rates['fwd'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "d69073eb",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "       zero       fwd       ytm\n",
      "0  0.050725       NaN  0.050725\n",
      "1  0.051401  0.051401  0.051395\n",
      "2  0.051051  0.050702  0.051054\n",
      "3  0.049987  0.047859  0.050026\n",
      "4  0.048496  0.044022  0.048607\n",
      "5  0.046704  0.039535  0.046931\n",
      "6  0.044738  0.034911  0.045127\n",
      "7  0.042717  0.030590  0.043311\n",
      "8  0.040675  0.026382  0.041514\n",
      "9  0.038674  0.022668  0.039791\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.legend.Legend at 0x7fa352270730>"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAY4AAAEWCAYAAABxMXBSAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAAsTAAALEwEAmpwYAABBUUlEQVR4nO3dd3wVVf7/8dfnppAGKSS0dAIKCBggQFCpUkIgQWyANNva1m2uu+r+3LXs+l1XXVdX167YEEERpAQFUQSUFnpvKZCEmkAoCSHl/P6YK4ZISSA3k/J5Ph55kCl35nOvJu+cmTPniDEGpZRSqrIcdheglFKqbtHgUEopVSUaHEoppapEg0MppVSVaHAopZSqEg0OpZRSVaLBoVQNEpEnReRju+tQ6nJocKgGT0QyRKRQRE6IyBERmSsi4XbXVRkiskhE7ra7DtWwaHAoZUkyxvgBLYEDwCs214OIuNtdg1LnosGhVDnGmFPA50CHn9aJiL+IfCgih0QkU0QeFxGHc9vtIrJURF5wtlbSRWRouddGi8j3InJcRBYAwec7t4j0E5EsEXlERPYDk0QkUETmOM99xPl9mHP/Z4DewKvO1tKrzvXtRGSBiOSJyHYRubXcORJFZIuznmwRebh6P0HVEGhwKFWOiPgAo4Dl5Va/AvgDrYG+wATgjnLbewLbsULhOeBdERHntk+A1c5tfwcmXqSEFkAQEAncg/UzOsm5HAEUAq8CGGP+H7AEeNAY42eMeVBEfIEFzvM2A8YAr4nIVc7jvwvca4xpDHQEvq3UB6NUOdoUVsoyU0RKAD/gIDAEQETcsIKkizHmOHBcRP4NjMf6JQyQaYx527n/B8BrQHMR8QS6AwONMUXAYhGZfZE6yoAnnPuDFRTTf9robGV8d4HXDwcyjDGTnMtrRGQ6cDOwGSgGOojIemPMEeDIRepR6he0xaGU5QZjTADQCHgQ+F5EWmC1FDyBzHL7ZgKh5Zb3//SNMabA+a0f0Ao4Yow5WeG1F3LIebkMsFpAIvKm8xLZMWAxEOAMtHOJBHqKyNGfvoCxWC0ZgJuARCDTeQmt10XqUeoXNDiUKscYU2qM+QIoBa4DDmP9lR5ZbrcIILsSh9sHBDovH5V/7QVLqLD8R+BKoKcxpgnQx7lezrP/XuB7Y0xAuS8/Y8z9AMaYVcaYEViXsWYC0yrxPpQ6iwaHUuWIZQQQCGw1xpRi/XJ9RkQai0gk8BBw0WcxjDGZQCrwlIh4ish1QFIVS2qMdbnqqIgEAU9U2H4A697LT+YAV4jIeBHxcH51F5H2zhrGioi/MaYYOIYVkEpViQaHUpbZInIC65fpM8BEY8xm57bfACeBNGAp1o3n9yp53Nuwbp7nYf3S/7CKdb0EeGO1fJYDX1XY/jJws7PH1X+d92EGA6OBHKzLaP/CugQH1r2ZDOdlr/uAcVWsRylEJ3JSSilVFdriUEopVSUaHEoppapEg0MppVSVaHAopZSqkgbx5HhwcLCJioqyuwyllKpTVq9efdgYE1JxfYMIjqioKFJTU+0uQyml6hQROedIB3qpSimlVJVocCillKoSDQ6llFJV0iDucSilGp7i4mKysrI4derUxXdu4Ly8vAgLC8PDw6NS+2twKKXqpaysLBo3bkxUVBQ/z6ulKjLGkJubS1ZWFtHR0ZV6jV6qUkrVS6dOnaJp06YaGhchIjRt2rRKLTMNDqVUvaWhUTlV/Zz0UlUtdezUKdbkpLFhfxrH076ktLQQL59W+DaJpknTNrTwb0GrJkGENQmiiZeX/oAopWqMBodNjDHkHMsnNXsXWw6lsftIJtknssk7nUOhOUiZ2xFEys5+UcF6KKDcRKUW9zI33Mu8cBhfxBFAI7cm+Lg1prFnE5o08iewkT9NffwJ8QmihV8grZo0JbRJU4J9/HA4tNGplCvMmDGDp5566qx1GzZsYO7cuQwdOtSmqqqHBocLlZaWse1QDmv37Wbr4XQy8vewvyCb/OJ9FMkhcDtx9gvKfPCSZjT3bEtLn1CiAyLpEBJF15ZtaWyKOJCzgdyD2zl2JI2TJ7MpPHWQU8VHOclpjjkcHHM4OOrmRl6pB0fd3MgpMZQVnr8+Y9yQMh/cjC8e4ouXmx/ebo1p7NGEJp5NCPDyJ8g7gBCfQFr4BdLCL4irW0Tj7dno/AdVSgEwcuRIRo4ceWb5rbfeYvLkyQwZMuSirzXGYIyptX/YaXBcppOni1iXk876/WnsyEtnz/G9HC7M4XjpAYodhxHH6TP7GiO4lQXg49aMFo26E+obRpugSDo2a01caBtCfAMveK4WITHn3lCQB3npkJcGebutf3N3Yw6lUXjqCMccDvLdHBx1uJHrFcQBryAOeDTmgMObQw4PchHyTQknS3LJL9lDTnEBcqrImguvAlPmjjfhRPheSdfmnRkU05240CtwSO38H1yp2mDHjh08/fTT/PjjjzgcDp5//nmmTZtGUVERI0eO5KmnniIjI4OhQ4fSv39/li1bxsyZM3n11VeZN28eIsLjjz/OqFGj7H4rgAZHpRw8cYzU7B1sOpDOrqOZ5JzIIrdoHyfLDvzikpIpc8fDBNPYvTkh3rFENA7niqaRXN08htiWrfHx9Kr+An2CrK+wbmetFsCn8Ag+eWm0yEuH3N0/h0vuDijIPfs4TUIhqDUExVLoH8l+7xCy3fzJEh/2FxZwsCCXbbk72Vuwne0nv2VHxld8mgGUeeHviKaNfwd6hsYytG0PIv1b6n0XVWs8NXszW3LO8ZfQZejQqglPJF110f2Ki4u57bbbeOGFF4iIiGD+/Pns3LmTlStXYowhOTmZxYsXExERwfbt25k0aRKvvfYa06dPZ926daxfv57Dhw/TvXt3+vTpQ8uWLav1fVwKDY4LGD/976zLT/nlJaVSHxoRQohnG1p4hxIVEEH74Gi6tIyhXUgYbg43ewo+F+9ACO1mfVVUeNQZJOW+cnfDtrl4FxwmGjjTq7txKytUOt4IXf9OQWkZC3dt5LuM1WzO3cyBoh2kHp3B6vzpvLYFHGVNaObZlvZBV9EnoivXt+5GoHdAjb1tpWqLv/71r1x11VWMHj0agPnz5zN//ny6dOkCwIkTJ9i5cycRERFERkYSHx8PwNKlSxkzZgxubm40b96cvn37smrVKpKTk217Lz/R4LiAiCahHD4VR6hfGDEBkVzVLJpuoW0JbRJkd2nVwzsAQrtaXxUVHoUjzstfuc5Q2b8R5j4Eq97FZ+izJHXoQ1KHnwMpJ/8Y83as5se969iZv4X9BbvZX7Ka7w5+yFOp4GmaEeZ9BVc368zA1nH0CO2El7sLWmBKVVCZloErLFq0iOnTp7NmzZoz64wxPPbYY9x7771n7ZuRkYGvr+9Z+9VWUpuLqy5xcXFGh1WvBsbA1lkw/3E4ugfaJ8Hgf0Bg1Dl3LyszbMjZx9c7V5O6fz0ZJ7ZRIOk4PJyXDIwDP0cY0Y3b0aNVLANbx9GuaVvcHfr3jLp8W7dupX379rad/8iRI3Tt2pVPPvmEXr16nVk/f/58/vrXv7Jw4UL8/PzIzs7Gw8ODgoIChg8fzqZNmwD44osvePPNN0lJSSEvL4+4uDhWrFhBixYtXFLvuT4vEVltjImruK9Lf0JFJAF4GXAD3jHGPFthuzi3J2J1NL3dGLPGuS0DOA6UAiU/FS8iQcBUIArIAG41xhxx5ftQTiLQYQS0HQzLXoUlL8KO+XDNg3DdQ9DI76zdHQ4hNqwVsWGtgCQAThaVsCRtN9+mr2bjoY3kFO5gQ8liNh6bz7vbQIwHQR6taRd4FdeGx9I3shvhjcP1fomqc9544w0OHjzI/ffff9b6xx57jNtuu+1MmPj5+fHxxx/j5nb2Je6RI0eybNkyrr76akSE5557zmWhUVUua3GIiBuwAxgEZAGrgDHGmC3l9kkEfoMVHD2Bl40xPZ3bMoA4Y8zhCsd9DsgzxjwrIo8CgcaYRy5Ui7Y4XORYDnzzJGyYCn4tYNBT0OlWqEIXQmMMOUcLWbhrC0v2rGH70c3kFu9GGmUjjhIA3PGlpdcVdAruSN+orvRoFUuwd7CL3pSqL+xucdQ1VWlxuDI4egFPGmOGOJcfAzDG/LPcPm8Ci4wxU5zL24F+xph9FwiO8vu0dL7+ygvVosHhYntXwrxHIGcNhHWHhH/9oodXVRSXlrE55wjf7N7Aypy1pB/fxknScTQ6eKYHWxO3MHq1vI6b2w2kW8uueDgqN6qnajg0OKqmtlyqCgX2llvOwmpVXGyfUGAfYID5ImKAN40xbzn3aW6M2QfgDI9m5zq5iNwD3AMQERFxmW9FXVB4D7h7IWz41GqBvDMArh4D1z8BTareddDDzUFseFNiw/sD/QHILyhmZeZ+vk1by5oDa8k6uY6vSj7j66xPccOb9gHdSGo7gIFRfWnmc87/JZRS1cSVwXGui9IVmzcX2udaY0yOMxgWiMg2Y8ziyp7cGTRvgdXiqOzr1CVyOCD2NuuG+ZJ/w7L/wZZZ0OePEP9r8Li83lP+Ph4Mah/OoPbhQDInikr4Zlsm07csYn3ucjYUb2DT0aX8cxWEeEYzIKIvw9oOoFNwJ73ZrlQ1c+VPVBYQXm45DMip7D7GmJ/+PSgiM4AewGLggIi0LHep6qCL6leXolFjGPgkdJ0A8/8KC5+G1R/AkGeg3XDrBns18Gvkzg1Xx3DD1TGUlt3Jmsw8Pt+Uyvd7F7Pv5EY+LfqQqbvex1P86BrSk+FtB9A77DqCvOpJV2qlbOTK4FgFtBWRaCAbGA3cVmGfWcCDIvIp1mWsfGcg+AIOY8xx5/eDgafLvWYi8Kzz3y9d+B7UpQpqDaMnw+7v4KvHYOo4iO4LCc9C8w7Veio3h9A9uindo4cAQ0g/fJKUTbuZs/N7MgpX82PxCpYfXAgIEb5XMiS6HwMi+9KhaQcdKkWpS+DS5zicvaZewuqO+54x5hkRuQ/AGPOGszvuq0ACVnfcO4wxqSLSGpjhPIw78Ikx5hnnMZsC04AIYA9wizEm70J16M1xm5WWwOpJ8O0/oOgYxN0F/f9iDZPiYkcLTvPttgPM3LqStQeXUeq1FYf3XkQMPm7+XBt6LYOi+nFNq2vwb+Tv8npUzdGb41VTK3pV1SYaHLVEQR4s+iesete6pNX/LxB3J7jVTI+o0yVlrEzPY+7mnSzMWMIxx0bcfHfgcC9AcHBFQEcGR/elT1gfrgy8Up8dqeNqQ3C4ubnRqVOnM8szZ84kKirKvoKAfv368cILLxAXd3Ye1JZeVUqdzScIEp+HbnfAV4/CvD9D6nvW5auY/i4/vae7g+vaBnNd22CMiWf7gePM35xDyo5VpJ1MZUvhdrYffYVX1r6Cv2dT+oX3oV94H+JbxuPn6XfxEyhVgbe3N+vWravy60pKSnB3v/xfz9V1nIo0OFTNa94BJnwJ21Pg67/ARzfAlcNg8N+h6XmGjq9mIkK7Fk1o16IJv72+HQeP3cK32w6SsnUnqw78yGHvrXxZOI8vd8/AgRuxIV3oF9GH3qG9iQmI0daIumTr1q3jvvvuo6CggJiYGN577z0CAwPp168f11xzDT/88APJycn873//Y/fu3eTn5xMUFMSiRYvo06cPvXv3ZtKkSeTl5fH73/+ewsJCvL29mTRpEldeeSXvv/8+c+fO5dSpU5w8eZK5c+dyxx13sGXLFtq3b09h4QUm6akkDQ5lDxFoNwzaDLS67i5+AV6Lh/gHoM/D1qWsGtSsiReje0QwukcEhaf78cOuw8zfks3C9FUcd2wi9dR21hx6kRdXv0iIdwv6h/ehT1gferXqhaebZ43Wqi7BvEetQTqrU4tOMPTZC+5SWFhIbGwsANHR0cyYMYMJEybwyiuv0LdvX/72t7/x1FNP8dJLLwFw9OhRvv/+ewAWLFjAli1bSE9Pp1u3bixZsoSePXuSlZVFmzZtOHbsGIsXL8bd3Z1vvvmGv/zlL0yfPh2AZcuWsWHDBoKCgnjxxRfx8fFhw4YNbNiwga5dzzGoaRVpcCh7uTeC3g9ZDwwufBp+eAnWT7EeHrx6TJWGL6ku3p5uDOzQnIEdmlNW1oUN2fl8s+UAX2/bRkbhGnL8tvFZwUym7ZiGt7svgyKvJyEqgfhW8foEuzpLxUtV+fn5HD16lL59+wIwceJEbrnlljPby0/U1Lt3bxYvXkx6ejqPPfYYb7/9Nn379qV79+5njjVx4kR27tyJiFBcXHzmtYMGDSIoyOp8snjxYn77298C0LlzZzp37nzZ70uDQ9UOTVrCyNeh+93w1SPw5QOw6m1r+JKIigMO1ByHQ4gNDyA2PICHh1zJ3rxBLNx6gAXbslm1bxXFjdcz+/QCZu2eRWMPfwZHDWRI1BC6t+iuDx7WJhdpGdQW5YdV7927N2+88QY5OTk8/fTTPP/882cuV4E1z0f//v2ZMWMGGRkZ9OvX75zHAar90qp2Yle1S1g3uHM+3Pg2HN8P7w2G6XdDfrbdlQEQHuTD7ddGM/mu61jxh1/z/3o8SevC5yjYO4Ejh6OZsWMO9yy4hwHTrucfy/9B6v5UykzZxQ+sGgR/f38CAwNZsmQJAB999NGZ1kdFPXv2PDPVrJeXF7Gxsbz55pv07t0bsFocoaGhALz//vvnPWefPn2YPHkyAJs2bWLDhg2X/T70TyJV+zgc0PlWuDIRlv4HfnwFts21hm6/5kHw8La7QgACfT0Z3yuK8b2iSDsUx8y12Uxfm86BknWUBWzks6IZTN0+lWY+zRgcOZiE6AQ6B3fWG+sN3AcffHDm5njr1q2ZNGnSOfdr1KgR4eHhZ2YE7N27N1OmTDnTvffPf/4zEydO5MUXX2TAgAHnPd/999/PHXfcQefOnYmNjaVHjx6X/R70OQ5V+x3JsIYv2ToL/COs3lcdRlTb8CXVqazMkJp5hBlrs5izMYNCj034BW3EeG+jjBJC/UIZHDWYhKgE2ge11xBxodrwHEddog8AVqDBUU+kL7aGLzmwCa4aCcmv/mLyqNrkVHEpC7ceZMbaLBbt3AO+mwkM2cJpj22UUUpE4wgSohNIiEqgbWBbu8utdzQ4qkaDowINjnqktAR+fNkaviT4Chj1MQTX/l+6uSeKmL0+hxlrs1m/LwfPxpsJbrGVE7IdQxkx/jFnQiTKP8rucusFDY6q0eCoQIOjHkpbBJ/fCSWn4YbXoEOy3RVV2q6DJ5ixNouZa3PIOX4Qn8AtNG2+laNl2zEY2gW1IyEqgSFRQwhrHGZ3uXWWBkfVaHBUoMFRT+VnwbQJkL0arv0dDPgbuNWd/h5lZYYV6XnMWJtFysb9nCzNpWmzrTQO3szh4p0AdAruxJCoIQyJGkIL39ox33RdocFRNRocFWhw1GMlRda4V6nvQVRvuHkS+IXYXVWVFZ4uZcHWA8xYk8XinYcpc8slPGwnHk02cPD0bgC6NuvKkKghDI4arHOuV4IGR9VocFSgwdEArJ0Mcx8C7yC49UMI7253RZfs0PEiZq3PYcbaLDZlH8O9US4xUTsp9VnHgVMZOMRB9+bdGRI9hIERAwn0CrS75FpJg6NqqhIc+gCgqh+6jIW7FlhDtE8aCivfhjr6R1FI40bcdV00c37Tm/l/6MPd8T04tr8fu9beB3sfJto9mYz8HJ5e9jT9p/Xnvm/uIyUthcKSyx+8TlUfYwzXXXcd8+bNO7Nu6tSpZx7ma9GiBaGhocTGxhIbG8vp06cREcaPH39m/5KSEkJCQhg+fLgdb+G86s4FYaUupmVnuPd7+OIeSHkYslJh+H/A08fuyi7ZFc0b8+jQdvxpyJWsSMtl+ppsvtrUnJOne9IiOI+oqB3syF3BI9mP4Ovhy+DIwSTFJNGteTed3dBmIsIbb7zBLbfcQv/+/SktLeXxxx9n8+bNxMTE8OSTT+Ln58fDDz985jW+vr5s2rTpzIi3CxYsOPN0eG2iwaHqF+9AGDMVFj9vTRp1YBOM+siayrYOc3MI17QJ5po2wfz9hqtYsOUA09dks3R1U8pMT66KySU4cCNfZ3zNjF0zaOXbiuExw0lqnaTde23UsWNHkpKS+Ne//sXJkyeZMGECMTEXnjpg6NChzJ07l5tvvpkpU6YwZsyYM0OU1BYaHKr+cTig3yMQ2tUa5+rNfnDjW3Blgt2VVQsfT3dGxIYyIjaUg8dOMX1NNlNW7mHz7hD8ffsT32EfRV4reGfjO7y14S06B3cmKSaJhKgEArwC7C7fFv9a+S+25W2r1mO2C2rHIz0eueh+TzzxBF27dsXT05PK3GsdPXo0Tz/9NMOHD2fDhg3ceeedGhxK1Zi2g6xLV1PHw5RR0OfP0O9RcLjZXVm1adbEi/v7xXBvn9b8uDuXKSv38PVqByVlI+kSfRNtoneyq3ARz6x4hn+t+hd9w/qSFJNEn9A+eNTQlL0Nna+vL6NGjcLPz49GjRpddP/OnTuTkZHBlClTSExMrIEKq06DQ9VvgVFw13yY+zAsfs565uOmd6xpbOsRh0POTIt76HgRn6/O4tNVe/js2xgCfNoxuHMJXoFr+WH/AhbuWYh/I3+GRg0lOSaZjsEd6/2YWZVpGbiSw+HAUYW5ZZKTk3n44YdZtGgRubm5Lqzs0mhwqPrPwxtGvGp10U35E7zZF0Z9CK262F2ZS4Q0bnSmFbIsLZdPVuzhy5X7KSmLpXt0XxLbHeaA+YEZu2bw6fZPiWoSRXJMMsNbD6elX0u7y1fAnXfeib+/P506dWLRokV2l/MLGhyqYRCBbrdb031OnQDvDoFhL0DXCXZX5jIOh3Btm2CubXN2K+S1eR4E+AwiKXYUoaE7WHFoAf9d+1/+u/a/dG/RnaTWSQyKHISfZ+0dQLK+CwsL43e/+53dZZyXPgCoGp6TuTD9Tmu8q64TYOjz4OFld1U1oqzMWK2QlXuYv3k/xaWGHtFBDI31pMBzFfMy5pB5LBMvNy8GRAwgOSaZ+JbxuNXB+0L6AGDV6JPjFWhwqF8oK4XvnoEl/4aWsVaX3YAIu6uqUYdPWK2QKSv3kJlbQICPBzd2CaXbFcdYe2Qh89Lncez0MUK8QxjWehhJMUlcEXiF3WVXmgZH1WhwVKDBoc5r21yYcZ/V0+qmd6DNQLsrqnHnbIVEBXFr9xb4BO7kq4w5LMlaQokpoV1QO4a3Hs6w1sNq/XhZGhxVU2uCQ0QSgJcBN+AdY8yzFbaLc3siUADcboxZU267G5AKZBtjhjvXPQn8Cjjk3O0vxpiUC9WhwaEuKHc3TB0HB7dC//8Hvf9oPQvSAJ27FRLG8C6N2XZ8CbN3z2ZT7ibcxI1erXqRHJNM//D+eLnXvkt9W7dupV27dvW+x1h1MMawbds2+4PD+Ut/BzAIyAJWAWOMMVvK7ZMI/AYrOHoCLxtjepbb/hAQBzSpEBwnjDEvVLYWDQ51UadPwuzfwcbP4IoEGPkmeAfYXZVtztcKua1nBO3CC5m/J4XZabPZf3I/fh5+DI4aTFLrJLo271prhjpJT0+ncePGNG3aVMPjAowx5Obmcvz4caKjo8/aZkdw9AKeNMYMcS4/5izyn+X2eRNYZIyZ4lzeDvQzxuwTkTDgA+AZ4CENDuVyxliDI379GPiHW/c9WnSyuyrbna8VMrpHKEfLtjNr9ywWZC6goKSAUL9QkmKSSG6dTHiTcFvrLi4uJisri1OnTtlaR13g5eVFWFgYHh5nPxRqR3DcDCQYY+52Lo8HehpjHiy3zxzgWWPMUufyQuARY0yqiHwO/BNoDDxcIThuB45hXcb6ozHmyDnOfw9wD0BERES3zMxMl7xPVQ/tWQGfTYTCo5D0Mlw9yu6KaoXztULG9AynX7sAluZ8x+zds1m+bzkGQ9dmXUmKSWJI1BAaeza2u3x1CewIjluAIRWCo4cx5jfl9pkL/LNCcPwZaAkkGmMeEJF+nB0czYHDgAH+DrQ0xtx5oVq0xaGq7PgBa2razKXQ/Vcw5P/A3dPuqmqNwyeKmO5shWTkFuDv7cFNXcMYGx+Br88J5qbNZdbuWaTlp9HIrRH9w/uTHJNMr1a9cHfo42N1RZ26VAX8FhgPlABeQBPgC2PMuArniALmGGM6XqgWDQ51SUpLYOGT8OMrENYdbvkA/GvfENd2KiszLE/LZXK5Vsi1bZoyrmck17dvxo6jW5m1exYp6SnkF+UT7B3MsOhhJLdJrlNdexsqO4LDHevm+PVANtbN8duMMZvL7TMMeJCfb47/1xjTo8Jx+nF2i6OlMWaf8/s/YF3+Gn2hWjQ41GXZPBO+/LU1dMnN70F0H7srqpUOHS9iWupePlmxh+yjhTRr3IjRPSIY0yOcYD93FmctZtbuWSzOWnyma29yTDKJ0Yk09W5qd/nqHOzqjpsIvITVHfc9Y8wzInIfgDHmDWd33FeBBKzuuHcYY1IrHKMfZwfHR0As1qWqDODen4LkfDQ41GU7tN3qspu7CwY+Cdf81hrGRP1CaZnhu20H+XhFJt/vOIRDhIHtmzEuPpJrY4LJP32UeenzmLV7FptzN+MmblwXeh3JMcn0C++Hp5teEqwt9AFADQ51uYqOWy2PLV9C+yQY8Rp4NbG7qlptT24Bk1dm8llqFnknTxMd7MvYnhHc3C2MAB9Pdh/dzazds5izew4HCw/SxLMJCVEJJLdJpnNwZ+1GazMNDg0OVR2MgWWvwoInrFkFx06r87ML1oSiklLmbdzPx8szSc08QiN3B8M7t2JcfASx4QGUmTJW7FvBrLRZLMxcyKnSU0Q1iSIpJomk1kk6aq9NNDg0OFR1Sl8C08aDwwPGTbfmO1eVsnXfMT5ensnMtdmcPF1Kx9AmjOsZSXJsK3w83Tlx+gQLMhcwa/csUg+kIgjdW3QnOSaZQZGD8PGou3PI1zUaHBocqrod2g4fjbQuYY2ZAlHX2V1RnXL8VDEz12bz8fI9bD9wnMZe7tzUNYxx8RG0aWY995F1PIs5aXOYtXsWe4/vxdvdm4ERA0luk0yPFj1qzVPq9ZUGhwaHcoX8LPjoRjiSATe/a937UFVijCE18wgfL89k3sb9nC4tI751EOPiIxncoQWe7g6MMaw7tI5Zu2fxdfrXHC8+TgvfFgxvPZzkmGSi/aMvfiJVZRocGhzKVQryYPItkLMGhr8E3SbaXVGddfjEz116s44UEuzXiDE9whnTI4JWAd4AnCo5xaKsRczaNYsfc36k1JTSKbgTyTHJDI0ein8jf5vfRf2hwaHBoVzp9EmYNgF2fQMD/mqNsKs9gi5ZaZlh8Y5DfLw8k2+3H0SAAe2aMy4+gj5tQ3A4rM/2cOHhM0+p7ziyA3eHO/3C+pEUk0Tv0N54uHlc+ETqgjQ4NDiUq5UWw8wHYOM06Hm/NUxJAx2evTrtzStgyso9TF21l9yTp4ls6sNtPSK4JS6cIN+fn/nYnredL3d/ydy0ueSdyiOgUQAJUQkkxSTRKbiTdu29BBocGhyqJpSVwfz/B8tfg063WM966BhX1aKopJSvNu1n8vI9rMzIw9PdwbBOLRkXH0nXiIAzwVBSVsKynGXM3j2bb/d+S1FpEVFNohjeejjDY4YT6qfDxlSWBocGh6opxsDS/8DCpyDmemt4dk9fu6uqV7bvP87kFZl8sSabE0UltG/ZhHHxEdwQG4pvo58HUTx++jjfZH7D7LTZrNq/CoC45nEkxSQxKHKQjtp7ERocGhyqpq3+AOb8Hlp1hbGfgU+Q3RXVOyeLSpi5zurSu3XfMfwauXNj11DGx0fStvnZoZBzIoc5aXOYvXs2GccyzozamxSTxDWtrtFRe89Bg0ODQ9lh6xxrePbASBj3BQTYO7lRfWWMYc2eo3y8PJO5G/ad6dI7Pj6KwVc1x8PNcda+mw5vYnbabOalz+No0VGCvIJIjE4kOSaZdkE63exPNDg0OJRdMpbClDHQqDGMnwEhV9pdUb2We6KIaalZfLw888wovWN6RDCmRwQt/M+eG724tJgl2UuYkzaHRXsXUVxWTJuANiTFJDEsehjNfZvb8yZqCQ0ODQ5lp30b4OOboKwYbvsMwrvbXVG9V1pmWLT9IB8t/3mU3iFXNWdcfCS9Wv9yHvL8ony+zvia2btns+7QOgShZ8ueJMUkMTBiYIMc6kSDQ4ND2S0v3Rqi5MQBuPUjaDvQ7ooajMzck0xesYdpqXs5WlBMm2Z+jI+PZGTXUJp4/fJZjz3H9pwZ6iT7RPaZoU6GxwynZ4ueuDncbHgXNU+DQ4ND1QYnDsLHN8LBrXDD69D5VrsralBOFZcyZ8M+PlqWwfqsfHw83bihi3UzvX3LXw6Rf9ZQJxlfc/z0cZp5N2NY62EkxSTRNrCtDe+i5mhwaHCo2uJUPnw6FjKWQMKzEH+/3RU1SOv3WjfTZ63PoaikjO5RgYyLj2Rox5Z4uv/ywc2i0iK+3/s9s9NmszRr6ZlZDJNaJ5HYOpFg72Ab3oVraXBocKjapPgUfHE3bJ0N1z0E1/9NhyixydGC03yWmsXHKzLJzC0g2M+T0d0jGNMzglDn+FgV5Z3KY176PObsnsOm3E24iRu9WvUiOSaZ/uH98XL3Oufr6hoNDg0OVduUlcLch2D1+9BlvDVAops+S2CXsjLD4p3W+FgLt1njYw1s35zxvawpb38aH6uitPw05uyew+y02ew/uR9fD18GRw5mWOthxDWPq9P3QzQ4NDhUbWQMfPcMLH4e2g2Hm94Bj3P/latqTsXxsaKDfRkXH8nNXcPw9zn3wIllpozVB1Yza/csFmQu4GTxSUK8Q0iITiAxOpGrml5V554P0eDQ4FC12Yo3Yd6fIfJaa1IoLx0avDb4acrbD5dlsGbPUbw8HIy4OpTxvSLpGHr+/0aFJYUszlpMSloKS7KXUFxWTETjCBJbJzI0eiit/evGdMMaHBocqrbb+DnMuA9C2sG4z6FxC7srUuVsys5n8opMZq7NobC4lC4RAYyPjySxU0u8PM5/OerY6WMszFzI3PS5rNy3EoOhfVB7EqMTSYhOoIVv7f3vrMGhwaHqgl0LYep48AuxnjIPqht/mTYk+YXFTF9tPZmedvgkQb6e3BoXztieEYQHXfghwUMFh/g642tS0lPYeHgjgtCteTeGRg9lcORgArwCauZNVJIGhwaHqiuyVsPkm8HhBuOmQ8ur7a5InYMxhh935/LhsgwWbDmAAQZc2YxxvSLpW26yqfPZc2wPKekppKSnkJ6fjru4c23otQyNHkr/8P614kl1DQ4NDlWXHNphPWV+Kt+65xHd2+6K1AXkHC3k05V7+GTlXg6fKCIiyIexPSO4uVsYTf0aXfC1xhi2H9lOSpoVIgcKDuDt7k2/8H4Mix7GNa2usW0mQw0ODQ5V1+RnW0+Z56XBTe9Ch2S7K1IXcbqkjK837+ej5ZmsTM/D083B0E4tGNszku5RgRftVVVmylh7cC0paSl8nfk1+UX5+DfyZ1DkIBKjE+nWvBsOqblZJTU4NDhUXVSQB5+MguxUGPYixN1hd0WqknYcOM4nK/YwfU0Wx0+VcEVzP8b2PP/4WBUVlxazbN8yUtJT+HbPtxSWFNLMpxlDo4aS2DqR9kHtXd6915bgEJEE4GXADXjHGPNshe3i3J4IFAC3G2PWlNvuBqQC2caY4c51QcBUIArIAG41xhy5UB0aHKpOO30Spk2EXQug/+PQ52F9yrwOKThdwpz1+5i8IpP1Wfl4e7iRfHUrxsVH0imsct2uC4oL+D7re1LSUlias5SSshKimkSRGG11743yj3JJ7TUeHM5f+juAQUAWsAoYY4zZUm6fROA3WMHRE3jZGNOz3PaHgDigSbngeA7IM8Y8KyKPAoHGmEcuVIsGh6rzSovhy1/DhqnQ415rjCtHzV2yUNVjY5bVpffLdVaX3s5h/oztGUHS1a3w8azcqAH5RfksyFxASnoKqftTMRiuanoVQ6OHkhCVUK1ziNgRHL2AJ40xQ5zLjwEYY/5Zbp83gUXGmCnO5e1AP2PMPhEJAz4AngEeKhcc5fdp6Xz9BWfG0eBQ9UJZGSz4Kyx7FTreBDe8Ae6edlelLsGxU8XMXJvNx8sz2XHgBI2dU96OjY/kiuaVnwf9wMkDfJXxFSnpKWzJ3YIgdG/RnaHRQxkUOQj/Rpf3IKkdwXEzkGCMudu5PB7oaYx5sNw+c4BnjTFLncsLgUeMMaki8jnwT6Ax8HC54DhqjAkod4wjxpjAc5z/HuAegIiIiG6ZmZkueZ9K1Shj4IeX4ZsnIGaANa9HIz+7q1KXyBjD6swjfLw8k5SN+zldWkaPqCDGxkeQ0LEFjdwrP85Ven4689LnkZKeQuaxTNwd7lwXeh0PXP0A7Zu2v6T6zhccrhxR7VwXYSum1Dn3EZHhwEFjzGoR6XcpJzfGvAW8BVaL41KOoVStIwLX/R58msLs38KHydaMgr5N7a5MXQIRIS4qiLioIP6WdJrPV+9l8oo9/O7TdQT5enJLtzDG9IggKtj3oseK9o/mgdgHuP/q+9mSt4WUtBS+Sv/KNXXXxktVwG+B8UAJ4AU0Ab4wxozTS1VKOW1Lgc/vAP9wmDAT/MPsrkhVg7Iyww+7DzN5+R4WbD1AaZmhd9tgxvaMZGD7Zri7Vf7eVmlZKQ5xXHLvKzsuVblj3Ry/HsjGujl+mzFmc7l9hgEP8vPN8f8aY3pUOE4/zr5U9TyQW+7meJAx5s8XqkWDQ9VbmT9a3XW9AqzwaBpjd0WqGh04doqpq/YyZeUe9uWfonmTRozqHsHo7uG0Os9cIdXJru64icBLWN1x3zPGPCMi9wEYY95wdsd9FUjA6o57hzEmtcIx+nF2cDQFpgERwB7gFmNM3oXq0OBQ9VrOOutBQXGzwqP5VXZXpKpZSWkZ320/xOQVmXy/4xACXN++OWN7RtCnEsObXCp9AFCDQ9Vnh7bDhyOguNAa3yrsFz/rqp74aa6Qaal7OXziNOFB3ozpEcGtceEEX2R4k6rS4NDgUPXdkQwrPE4cgts+heg+dlekXOin4U0mr8hkeVoeHm5CQseWjO0ZQc/ooGp5qlyDQ4NDNQTH9lmDI+alwa0fwJVD7a5I1YBdB0/wyYo9fL56L8dOldCmmR9je0ZwY5fzz1hYGRocGhyqoSjIs+557NsAI9+EzrfYXZGqIYWnS5mzIYfJK/awbq81Y+HrY7vRv12zSzqeHc9xKKXs4BMEE2bBlDHwxa/g9HGIu9PuqlQN8PZ045a4cG6JC2dTdj6frNxT6fGwqkIHu1GqPvJqYk0/23YwzPkDLH3J7opUDesY6s//jexU7TfMQYNDqfrLwxtGT7bGtfrmCVj4tDVkiVKXSS9VKVWfuXnAjW+Dpx8s+TecOgZDn9ORddVl0eBQqr5zuEHSy9blqx9fgaLjMOJ/4KY//urS6P85SjUEIjDo79DIH777B5w+ATe/B+7Vf/1b1X/aXlWqoRCBvn+yLlVtmwOf3GrNLqhUFWlwKNXQ9LwXbngd0hfDhzdA4VG7K1J1jAaHUg1R7G1wyweQsxbeH24NU6JUJVUpOEQkXkS+FZEfROQGF9WklKoJHZLhtqmQuwsmJUB+lt0VqTrigsEhIi0qrHoISMYaBv3vripKKVVD2lxvDcV+4iC8lwCHd9ldkaoDLtbieENE/ioiXs7lo8BtwCjgmCsLU0rVkIh4uH0OFBdYLY/9m+yuSNVyFwwOY8wNwDpgjoiMB34PlAE+wA2uLU0pVWNaXg13fAVunvB+IuxdZXdFqha76D0OY8xsYAgQAHwBbDfG/NcYo3fTlKpPQq6AO78Cn6bWvB5pi+yuSNVSF7vHkSwiS4FvgU3AaGCkiEwREZ3cWKn6JiDCankERsLkW2DbXLsrUrXQxVoc/8BqbdwE/MsYc9QY8xDwN+AZVxenlLJB4+Zw+1xo0QmmjocN0+yuSNUyFwuOfKxWxmjg4E8rjTE7jTGjXVmYUspGPkEw4UuIvAa+uAdWvWN3RaoWuVhwjMS6EV6C1ZtKKdVQNGoMYz+HKxJg7h9hyYt2V6RqiQsOcmiMOQy8UkO1KKVqGw8vGPURzLgPFj4FRcfg+iesca9Ug6Wj4yqlLszNA258yxqWfel/rDk9El/QOT0aMA0OpdTFOdxg2IvW5asfXrbm9LjhNStUVIOjwaGUqhwRGPQ0ePlb09CePmnN6eHhdfHXqnrFpW1NEUkQke0isktEHj3HdhGR/zq3bxCRrs71XiKyUkTWi8hmEXmq3GueFJFsEVnn/Ep05XtQSlXQ+4/Wpartc605PYpO2F2RqmEuCw4RcQP+BwwFOgBjRKRDhd2GAm2dX/cArzvXFwEDjDFXA7FAgojEl3vdf4wxsc6vFFe9B6XUefT4FYx8EzKWwkc3QOERuytSNciVLY4ewC5jTJox5jTwKTCiwj4jgA+NZTkQICItncs//Rnj4fwyLqxVKVVVV4+GWz+Afeudc3ocvPhrVL3gyuAIBfaWW85yrqvUPiLiJiLrsB48XGCMWVFuvwedl7beE5HAc51cRO4RkVQRST10SIfVUsol2ifBbdMgLw3eHQy5u+2uSNUAVwbHuTp6V2w1nHcfY0ypMSYWCAN6iEhH5/bXgRisS1j7gH+f6+TGmLeMMXHGmLiQkJCqV6+UqpyY/jBhFpzKh3cH6ci6DYArgyMLCC+3HAbkVHUfY8xRYBHW5FEYYw44Q6UMeBvrkphSyk7h3eHub6BRE/ggSQdHrOdcGRyrgLYiEi0inljjXc2qsM8sYIKzd1U8kG+M2SciISISACAi3sBAYJtzuWW514/EGrVXKWW3pjFw1wJofhV8OhZWvGV3RcpFXPYchzGmREQeBL4G3ID3jDGbReQ+5/Y3gBQgEdgFFAB3OF/eEvjA2TPLAUwzxsxxbntORGKxLmllAPe66j0oparILwQmzobpd8O8P0H+Hhj4tD5lXs+IMfW/s1JcXJxJTU21uwylGo6yUpj3CKx6G666EW54XR8UrINEZLUxJq7ien1yXClV/RxukPg8BITDgr/BiQMw6mNruHZV52n7USnlGiJw7e/gpnchaxW8NwSOZNpdlaoGGhxKKdfqdDOMn2m1Ot4dBDnr7K5IXSYNDqWU60VdC3fOBzdPmJQIOxfYXZG6DBocSqma0ayd9axH0xj4ZBSs/sDuitQl0uBQStWcxi3gjhTrafPZv4Vv/wENoGdnfaPBoZSqWY0aw5hPoct4WPw8zLwfSk7bXZWqAu2Oq5SqeW4ekPwKBETAd8/A8X1w64fWJFGq1tMWh1LKHiLQ98/Ww4EZS+G9oZCfbXdVqhI0OJRS9oq9DcZ+Bkf3wDsD4cBmuytSF6HBoZSyX8wAuHMeYOC9BEhbZHdF6gI0OJRStUOLTlZ3Xf8w+PhmWP+p3RWp89DgUErVHv5hcMc8iIiHGfdava60u26to8GhlKpdvANg3BfQ6VbrOY/Zv4PSErurUuVod1ylVO3j7gk3vmWNrrvk31Z33ZsnQSM/uytTaItDKVVbicD1f4PhL8Gub+D9YXD8gN1VKTQ4lFK1Xdwd1pPmh3fAuwPh0A67K2rwNDiUUrXfFUPg9rlQXGgNzZ75o90VNWgaHEqpuiG0q9Vd1zcEPrwBNn1hd0UNlgaHUqruCIyCu+ZDqy7w+R3w4yvaXdcGGhxKqbrFJwgmfAkdRsD8x2HeI1BWandVDYoGh1Kq7vHwgpvfh/hfw8o3YdoE6/6HqhEaHEqpusnhgIT/g4RnYdtc+CAJTh62u6oGQYNDKVW3xd9vzeWxf6PV4yp3t90V1XsaHEqpuq9DMkyYBYVHraHZ9yy3u6J6zaXBISIJIrJdRHaJyKPn2C4i8l/n9g0i0tW53ktEVorIehHZLCJPlXtNkIgsEJGdzn8DXfkelFJ1RERPq7uud6B12Wrj53ZXVG+5LDhExA34HzAU6ACMEZEOFXYbCrR1ft0DvO5cXwQMMMZcDcQCCSIS79z2KLDQGNMWWOhcVkopaBpjhUdYd5h+F3z/nHbXdQFXtjh6ALuMMWnGmNPAp8CICvuMAD40luVAgIi0dC6fcO7j4fwy5V7zgfP7D4AbXPgelFJ1jU8QjJ8BnUdZ85nPvB9Kiuyuql5xZXCEAnvLLWc511VqHxFxE5F1wEFggTFmhXOf5saYfQDOf5ud6+Qico+IpIpI6qFDhy73vSil6hL3RjDyTej3F1g/BT4aCQV5dldVb7gyOOQc6yq2Gc+7jzGm1BgTC4QBPUSkY1VObox5yxgTZ4yJCwkJqcpLlVL1gQj0ewRufAeyVlk3zbXHVbVwZXBkAeHllsOAnKruY4w5CiwCEpyrDohISwDnvwerrWKlVP3T+RZnj6sjVnhkLrO7ojrPlcGxCmgrItEi4gmMBmZV2GcWMMHZuyoeyDfG7BOREBEJABARb2AgsK3cayY6v58IfOnC96CUqg8ie1k3zX2C4MNk2DDN7orqNJcFhzGmBHgQ+BrYCkwzxmwWkftE5D7nbilAGrALeBt4wLm+JfCdiGzACqAFxpg5zm3PAoNEZCcwyLmslFIX1jQG7loAYT3gi1/Bome1x9UlEtMAPri4uDiTmppqdxlKqdqg5DTM/q1107zzKEh+xbqZrn5BRFYbY+Iqrtc5x5VSDYu7J9zwOgTFwHf/gKN7YfRk6zKWqhQdckQp1fCIQN8/wU3vQvZq7XFVRRocSqmGq9PNMHEWnDoK71wPGT/YXVGdoMGhlGrYIuKdPa6C4cMRsH6q3RXVehocSikV1BruXmCFyIx74Lt/ao+rC9DgUEopsEbVHfcFxI6F75+1uuwWn7K7qlpJe1UppdRP3D1hxP+sFsi3f3f2uPoEfJvaXVmtoi0OpZQqTwT6PAw3vwc5a62b5od32l1VraLBoZRS59LxJpg4G4qOWd1105fYXVGtocGhlFLnE9ET7l4Ifs2sodnXTbG7olpBg0MppS4kKBrumm8NlDjzPvj2Hw2+x5UGh1JKXYx3IIydDl3GweLnrWlpG3CPK+1VpZRSleHuCcmvWj2uFj4N+VnOHlfBdldW47TFoZRSlSUCvf8IN0+CnHVWj6tDO+yuqsZpcCilVFV1vBFunwtFJ+DdgZC+2O6KapQGh1JKXYrw7vCrheDXwupxtXay3RXVGA0OpZS6VIFRzh5X18KXD1j3PsrK7K7K5TQ4lFLqcngHwLjp0GU8LPm3s8dVod1VuZT2qlJKqcvl5mFNQdu0DXzzhLPH1WTrwcF6SFscSilVHUTgut/DrR/C/g3wRu96OzGUBodSSlWnDiPgrgXg6QMfJMGSF+vdfQ8NDqWUqm4tO8M930OHZFj4FEwZBQV5dldVbTQ4lFLKFbyaWA8KJr4AaYvgjetgzwq7q6oWGhxKKeUqItDjV1aXXYc7vJ8IP75S5wdJ1OBQSilXa9UF7l0MVyTA/Mfh09ug8IjdVV0yDQ6llKoJ3gEw6mMY8k/YOR/e7APZq+2u6pK4NDhEJEFEtovILhF59BzbRUT+69y+QUS6OteHi8h3IrJVRDaLyO/KveZJEckWkXXOr0RXvgellKo2ItDrAbjjK+ty1btDYMWbde7SlcuCQ0TcgP8BQ4EOwBgR6VBht6FAW+fXPcDrzvUlwB+NMe2BeODXFV77H2NMrPMrxVXvQSmlXCK8u3Xpqs31MO/P8NlEOJVvd1WV5soWRw9glzEmzRhzGvgUGFFhnxHAh8ayHAgQkZbGmH3GmDUAxpjjwFYg1IW1KqVUzfIJgtFTYOBTsHUOvNkX9q23u6pKcWVwhAJ7yy1n8ctf/hfdR0SigC5A+X5sDzovbb0nIoHnOrmI3CMiqSKSeujQoUt8C0op5UIOh/W0+e1zoaQI3hkEq96t9ZeuXBkcco51FT+NC+4jIn7AdOD3xphjztWvAzFALLAP+Pe5Tm6MecsYE2eMiQsJCali6UopVYMie8F9SyDqOpj7EEy/G4qO213VebkyOLKA8HLLYUBOZfcREQ+s0JhsjPnipx2MMQeMMaXGmDLgbaxLYkopVbf5BsPYz2HA47D5C3irHxzYbHdV5+TK4FgFtBWRaBHxBEYDsyrsMwuY4OxdFQ/kG2P2iYgA7wJbjTEvln+BiLQstzgS2OS6t6CUUjXI4YA+f4IJs6wWx9sDYO3Hdlf1Cy4LDmNMCfAg8DXWze1pxpjNInKfiNzn3C0FSAN2YbUeHnCuvxYYDww4R7fb50Rko4hsAPoDf3DVe1BKKVtE94b7lkJ4D/jy1zDjfjh90u6qzhBTy2/CVIe4uDiTmppqdxlKKVU1ZaXw/XPw/b8g5EpryPaQK2vs9CKy2hgTV3G9PjmulFK1lcMN+j8G47+Ak4et+x7rp9pdlQaHUkrVejEDrEtXLWNhxj0w6ze2Tk+rwaGUUnVBk5YwcTZc9xCs+RDeGQiHd9lSigaHUkrVFW7uMPAJuO0zOJYNb/WFTdNrvAwNDqWUqmuuGGxdumrWAT6/E+b+0XryvIZocCilVF3kHwZ3pECvB2HVO/DuIMhLr5FTa3AopVRd5eYBQ56xBks8kmENlLh1tstPq8GhlFJ1XbtEuHcJNI2BqeNg3qNQctplp9PgUEqp+iAwEu78CnrcCyteh0kJcHSPS06lwaGUUvWFeyNIfA5u+QAO74Q3ekPG0mo/jQaHUkrVN1fdAPcsglZdIDCq2g/vXu1HVEopZb+mMTBhpksOrS0OpZRSVaLBoZRSqko0OJRSSlWJBodSSqkq0eBQSilVJRocSimlqkSDQymlVJVocCillKoSMcbYXYPLicghINPuOi5TMHDY7iJqEf08fqafxdn08zjb5XwekcaYkIorG0Rw1AcikmqMibO7jtpCP4+f6WdxNv08zuaKz0MvVSmllKoSDQ6llFJVosFRd7xldwG1jH4eP9PP4mz6eZyt2j8PvcehlFKqSrTFoZRSqko0OJRSSlWJBkctJyLhIvKdiGwVkc0i8ju7a7KbiLiJyFoRmWN3LXYTkQAR+VxEtjn/H+lld012EZE/OH9GNonIFBHxsrummiQi74nIQRHZVG5dkIgsEJGdzn8Dq+NcGhy1XwnwR2NMeyAe+LWIdLC5Jrv9DthqdxG1xMvAV8aYdsDVNNDPRURCgd8CccaYjoAbMNreqmrc+0BChXWPAguNMW2Bhc7ly6bBUcsZY/YZY9Y4vz+O9Ysh1N6q7CMiYcAw4B27a7GbiDQB+gDvAhhjThtjjtpalL3cAW8RcQd8gByb66lRxpjFQF6F1SOAD5zffwDcUB3n0uCoQ0QkCugCrLC5FDu9BPwZKLO5jtqgNXAImOS8dPeOiPjaXZQdjDHZwAvAHmAfkG+MmW9vVbVCc2PMPrD+CAWaVcdBNTjqCBHxA6YDvzfGHLO7HjuIyHDgoDFmtd211BLuQFfgdWNMF+Ak1XQpoq5xXrsfAUQDrQBfERlnb1X1lwZHHSAiHlihMdkY84Xd9djoWiBZRDKAT4EBIvKxvSXZKgvIMsb81AL9HCtIGqKBQLox5pAxphj4ArjG5ppqgwMi0hLA+e/B6jioBkctJyKCdQ17qzHmRbvrsZMx5jFjTJgxJgrrxue3xpgG+1elMWY/sFdErnSuuh7YYmNJdtoDxIuIj/Nn5noaaEeBCmYBE53fTwS+rI6DulfHQZRLXQuMBzaKyDrnur8YY1LsK0nVIr8BJouIJ5AG3GFzPbYwxqwQkc+BNVg9EdfSwIYeEZEpQD8gWESygCeAZ4FpInIXVrjeUi3n0iFHlFJKVYVeqlJKKVUlGhxKKaWqRINDKaVUlWhwKKWUqhINDqWUUlWiwaFUFYiIEZGPyi27i8ihi43UKyKxIpJ4iedMcY6CGyAiD1zKMZSqThocSlXNSaCjiHg7lwcB2ZV4XSxQpeAQi8MYk+gcvDAA0OBQttPgUKrq5mGN0AswBpjy0wYR6SEiPzoHHfxRRK50Ppz3NDBKRNaJyCgReVJEHi73uk0iEuX82ioir2E9zBYuIhkiEoz1MFeM8xjPi8hHIjKi3DEmi0hyDbx/1cBpcChVdZ8Co50TBXXm7NGKtwF9nIMO/g34P2PMaef3U40xscaYqRc5/pXAh8aYLsaYzHLrHwV2O4/xJ6yh5e8AEBF/rLGZdEQB5XI65IhSVWSM2eAc4n4Mv/xF7Q98ICJtAQN4XMIpMo0xyytRx/ci8j8RaQbcCEw3xpRcwvmUqhJtcSh1aWZhzf8wpcL6vwPfOWehSwLON31pCWf//JXf72QV6vgIGIvV8phUhdcpdcm0xaHUpXkPa7KgjSLSr9x6f36+WX57ufXHgcblljOA4QAi0hVrHomLqXgMsKYLXQnsN8ZsrlTlSl0mbXEodQmMMVnGmJfPsek54J8i8gPWvNc/+Q7o8NPNcaz5VYKcIx7fD+yoxDlzgR+cN9Kfd647gDV8uLY2VI3R0XGVqsNExAfYCHQ1xuTbXY9qGLTFoVQdJSIDsXpxvaKhoWqStjiUUkpVibY4lFJKVYkGh1JKqSrR4FBKKVUlGhxKKaWqRINDKaVUlfx/UU7fcNRcxFgAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#h)\n",
    "\n",
    "#creating a new column on the rates DataFrame to include the Yield to Maturities obtained\n",
    "rates['ytm']=yields\n",
    "\n",
    "#Visualizing the complete DataFrame\n",
    "print(rates)\n",
    "\n",
    "\n",
    "#Plot the three rates obtained\n",
    "plt.plot(range(1,11),rates[['zero','fwd','ytm']])\n",
    "plt.title('Bond rates')\n",
    "plt.xlabel('Maturity')\n",
    "plt.ylabel('%')\n",
    "plt.legend(['Zero','Forward','YTM'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "ae4a830f",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Exercise 2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "37b693e0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Close</th>\n",
       "      <th>Volume</th>\n",
       "      <th>Name</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Date</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2012-01-03</th>\n",
       "      <td>14.686786</td>\n",
       "      <td>302220800</td>\n",
       "      <td>AAPL</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2012-01-04</th>\n",
       "      <td>14.765714</td>\n",
       "      <td>260022000</td>\n",
       "      <td>AAPL</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2012-01-05</th>\n",
       "      <td>14.929643</td>\n",
       "      <td>271269600</td>\n",
       "      <td>AAPL</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2012-01-06</th>\n",
       "      <td>15.085714</td>\n",
       "      <td>318292800</td>\n",
       "      <td>AAPL</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2012-01-09</th>\n",
       "      <td>15.061786</td>\n",
       "      <td>394024400</td>\n",
       "      <td>AAPL</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-10-08</th>\n",
       "      <td>785.489990</td>\n",
       "      <td>16711100</td>\n",
       "      <td>TSLA</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-10-11</th>\n",
       "      <td>791.940002</td>\n",
       "      <td>14200300</td>\n",
       "      <td>TSLA</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-10-12</th>\n",
       "      <td>805.719971</td>\n",
       "      <td>22020000</td>\n",
       "      <td>TSLA</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-10-13</th>\n",
       "      <td>811.080017</td>\n",
       "      <td>14120100</td>\n",
       "      <td>TSLA</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-10-14</th>\n",
       "      <td>818.320007</td>\n",
       "      <td>12247200</td>\n",
       "      <td>TSLA</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>12315 rows × 3 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                 Close     Volume  Name\n",
       "Date                                   \n",
       "2012-01-03   14.686786  302220800  AAPL\n",
       "2012-01-04   14.765714  260022000  AAPL\n",
       "2012-01-05   14.929643  271269600  AAPL\n",
       "2012-01-06   15.085714  318292800  AAPL\n",
       "2012-01-09   15.061786  394024400  AAPL\n",
       "...                ...        ...   ...\n",
       "2021-10-08  785.489990   16711100  TSLA\n",
       "2021-10-11  791.940002   14200300  TSLA\n",
       "2021-10-12  805.719971   22020000  TSLA\n",
       "2021-10-13  811.080017   14120100  TSLA\n",
       "2021-10-14  818.320007   12247200  TSLA\n",
       "\n",
       "[12315 rows x 3 columns]"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#a)\n",
    "\n",
    "#setting start and end dates\n",
    "start = datetime.datetime(2012,1,1)\n",
    "end = datetime.datetime(2021,10,15)\n",
    "\n",
    "#choosing 5 stocks\n",
    "symbols =['AAPL','MSFT','NFLX','AMZN','TSLA']\n",
    "\n",
    "#creating an empty DataFrame to store the information\n",
    "stock_data = pd.DataFrame()\n",
    "\n",
    "#Loop through each stock given and download information available on each, on yahoo finance\n",
    "for i in symbols:\n",
    "    stock = []\n",
    "    stock = yf.download(i,start=start, end=end, progress=False)\n",
    "        \n",
    "    if len(stock) == 0:\n",
    "        None\n",
    "    else:\n",
    "        stock['Name']=i\n",
    "        stock_data = stock_data.append(stock,sort=False)\n",
    "\n",
    "#Deleting non-essencial information\n",
    "del stock_data['Open'] \n",
    "del stock_data['High']\n",
    "del stock_data['Low']\n",
    "del stock_data['Adj Close']\n",
    "\n",
    "#Visualizing the DataFrame\n",
    "stock_data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "405bfc54",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0, 0.5, '%')"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#b)\n",
    "#Filtering data and setting Date as index\n",
    "stock_data_prices = stock_data.pivot_table(index=['Date'], columns='Name', values=['Close'])\n",
    "stock_data_prices.columns = [col[1] for col in stock_data_prices.columns.values]\n",
    "\n",
    "#Calculating the cummulative returns\n",
    "cum_return = ((stock_data_prices.iloc[-1]-stock_data_prices.iloc[0]) / stock_data_prices.iloc[0]) * 100\n",
    "\n",
    "#plotting the cummulative returns for each stock\n",
    "plt.bar(symbols,cum_return)\n",
    "plt.title('Cumulative Returns')\n",
    "plt.xlabel('Stocks')\n",
    "plt.ylabel('%')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "f8773962",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>AAPL</th>\n",
       "      <th>AMZN</th>\n",
       "      <th>MSFT</th>\n",
       "      <th>NFLX</th>\n",
       "      <th>TSLA</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>AAPL</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.441520</td>\n",
       "      <td>0.560642</td>\n",
       "      <td>0.260841</td>\n",
       "      <td>0.316462</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>AMZN</th>\n",
       "      <td>0.441520</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.535481</td>\n",
       "      <td>0.414321</td>\n",
       "      <td>0.320874</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>MSFT</th>\n",
       "      <td>0.560642</td>\n",
       "      <td>0.535481</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.344906</td>\n",
       "      <td>0.332121</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>NFLX</th>\n",
       "      <td>0.260841</td>\n",
       "      <td>0.414321</td>\n",
       "      <td>0.344906</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.268085</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>TSLA</th>\n",
       "      <td>0.316462</td>\n",
       "      <td>0.320874</td>\n",
       "      <td>0.332121</td>\n",
       "      <td>0.268085</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          AAPL      AMZN      MSFT      NFLX      TSLA\n",
       "AAPL  1.000000  0.441520  0.560642  0.260841  0.316462\n",
       "AMZN  0.441520  1.000000  0.535481  0.414321  0.320874\n",
       "MSFT  0.560642  0.535481  1.000000  0.344906  0.332121\n",
       "NFLX  0.260841  0.414321  0.344906  1.000000  0.268085\n",
       "TSLA  0.316462  0.320874  0.332121  0.268085  1.000000"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#c)\n",
    "\n",
    "#Calculating daily returns\n",
    "daily_returns = stock_data_prices.pct_change()\n",
    "\n",
    "#Dropping the first line (Daily return N/A)\n",
    "daily_returns = daily_returns[1:]\n",
    "\n",
    "#Creating the correlation Matrix from the daily returns\n",
    "corr_df = daily_returns.dropna().corr(method='pearson')\n",
    "corr_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "ff84235e",
   "metadata": {},
   "outputs": [],
   "source": [
    "#d)\n",
    "#Saving the data onto 5 different csv files, one for each ticker\n",
    "for idx, chunk in enumerate(np.array_split(stock_data, 5)):\n",
    "    chunk.to_csv(f'/Users/guilhermemiranda/CF_{idx}.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "d41df1fc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Date</th>\n",
       "      <th>Close</th>\n",
       "      <th>Name</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2012-01-03</td>\n",
       "      <td>5.616000</td>\n",
       "      <td>TSLA</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2012-01-04</td>\n",
       "      <td>5.542000</td>\n",
       "      <td>TSLA</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2012-01-05</td>\n",
       "      <td>5.424000</td>\n",
       "      <td>TSLA</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2012-01-06</td>\n",
       "      <td>5.382000</td>\n",
       "      <td>TSLA</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2012-01-09</td>\n",
       "      <td>5.450000</td>\n",
       "      <td>TSLA</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12310</th>\n",
       "      <td>2021-10-08</td>\n",
       "      <td>294.850006</td>\n",
       "      <td>MSFT</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12311</th>\n",
       "      <td>2021-10-11</td>\n",
       "      <td>294.230011</td>\n",
       "      <td>MSFT</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12312</th>\n",
       "      <td>2021-10-12</td>\n",
       "      <td>292.880005</td>\n",
       "      <td>MSFT</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12313</th>\n",
       "      <td>2021-10-13</td>\n",
       "      <td>296.309998</td>\n",
       "      <td>MSFT</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12314</th>\n",
       "      <td>2021-10-14</td>\n",
       "      <td>302.750000</td>\n",
       "      <td>MSFT</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>12315 rows × 3 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "             Date       Close  Name\n",
       "0      2012-01-03    5.616000  TSLA\n",
       "1      2012-01-04    5.542000  TSLA\n",
       "2      2012-01-05    5.424000  TSLA\n",
       "3      2012-01-06    5.382000  TSLA\n",
       "4      2012-01-09    5.450000  TSLA\n",
       "...           ...         ...   ...\n",
       "12310  2021-10-08  294.850006  MSFT\n",
       "12311  2021-10-11  294.230011  MSFT\n",
       "12312  2021-10-12  292.880005  MSFT\n",
       "12313  2021-10-13  296.309998  MSFT\n",
       "12314  2021-10-14  302.750000  MSFT\n",
       "\n",
       "[12315 rows x 3 columns]"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#e)\n",
    "\n",
    "#Creating a variable with the path where the files are saved\n",
    "path = r'/Users/guilhermemiranda/'\n",
    "all_files = glob.glob(path + \"CF_*.csv\")\n",
    "\n",
    "list = []\n",
    "\n",
    "#Looping through all files and, for each, read it and append it to the empty DataFrame previously created\n",
    "for filename in all_files:\n",
    "    df = pd.read_csv(filename, index_col=None, header=0)\n",
    "    list.append(df)\n",
    "stock_data_imp = pd.concat(list, axis=0, ignore_index=True)\n",
    "\n",
    "#Deleting the volume variable from the DataFrame as we are only interested on the prices for each ticker\n",
    "del stock_data_imp['Volume'] \n",
    "\n",
    "#Visualizing the DataFrame\n",
    "stock_data_imp"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "519bff9a",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
