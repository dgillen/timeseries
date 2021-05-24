# Week 3 `Practitioner Packages`
![image](anomalies.png)

This week we'll focus on various time series packages that help a data science practioner more easily apply models, algorithms, transformation to time series.

As a practitioner, it's important to know what the latest/helper packages are being used in the industry in order to make you more efficient as a data scientist.  This week, we'll look at 3 of those packages while also looking at some side topics related to time series such as unevenly spaces timestamps as well as anomaly detection 

1. **[Darts](https://github.com/unit8co/darts)**: Easy manipulation and forecasting of time series
2. **[Slices](https://github.com/datascopeanalytics/traces)**: Unevenly-spaced time series analysis.
3. **[tssmoothie](https://github.com/cerlymarco/tsmoothie)** : Time-series smoothing and outlier detection in a vectorized way.
4. **[tslearn](https://github.com/rtavenar/tslearn)**:provides machine learning tools for the analysis of time series. 
5. **[pyts](https://github.com/johannfaouzi/pyts)**: aims to make time series classification easily accessible by providing preprocessing and utility tools, and implementations of state-of-the-art algorithms. Most of these algorithms transform time series, thus pyts provides several tools to perform these transformations.

## Lesson Plan

Read through the markdown, code, and outputs in the following notebooks in order to 
1. [Anomaly Detection](./les1-anomaly.ipynb): This notebook guides you through various types of anomalies & how to detect them using various smoothing methods from the `tssmoothie` pkg
2. [UnEvenly Spaced Time Series](./les2-traces.ipynb): Leverages the the `traces` package to handle uneven time series 
3. [Darts Tutorial](./les3-darts.ipynb): Step througt the execellent introductory tutorial to `darts`

## Homework

Write a **one-page brief** on the above packages in which you describe how a practitioner might apply these tools to various problems. Discuss the following topics

* **Importance**: Why is it important for a practitioner to be aware of these packages and packages like them?
* **Continued Education**: Besides this week's lesson, how else might you learn about these packes and others?
* **Challenges**: What are some key challenges in applying these packages?
* **Additional Packages** Review [MaxBenChrist's excellent summary of time series packages](https://github.com/MaxBenChrist/awesome_time_series_in_python) & select a few that look most promising to try next and state why?

## Course Objectives Addressed

By learning to apply the above python packages to various times series analysis, we've addressed the **11th Course Objective**: Apply various python packages to solve tangential time series problems such as anomaly detection


