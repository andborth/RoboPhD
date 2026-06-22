<code>
slopes = np.array([
    np.polyfit(df1.loc[df1[col].notna(), "Time"], df1.loc[df1[col].notna(), col], 1)[0]
    for col in df1.columns
    if col != "Time"
])
</code>