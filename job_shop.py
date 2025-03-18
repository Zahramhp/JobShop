import streamlit as st
import pandas as pd
from pyomo.environ import *
from pyomo.opt import SolverFactory

def solve_job_shop(jobs, machines, processing_times):
    model = ConcreteModel()
    
    # Sets
    model.J = RangeSet(len(jobs))  # Jobs
    model.M = RangeSet(len(machines))  # Machines
    
    # Decision Variables
    model.x = Var(model.J, model.M, domain=Binary)  # Whether job j is processed on machine m
    model.Cmax = Var(domain=NonNegativeReals)  # Makespan
    
    # Constraints
    def one_machine_per_job_rule(model, j):
        return sum(model.x[j, m] for m in model.M) == 1
    model.one_machine_per_job = Constraint(model.J, rule=one_machine_per_job_rule)
    
    def machine_capacity_rule(model, m):
        return sum(model.x[j, m] for j in model.J) <= 1
    model.machine_capacity = Constraint(model.M, rule=machine_capacity_rule)
    
    def makespan_rule(model, j, m):
        return model.Cmax >= sum(model.x[j, m] * processing_times[j-1][m-1] for j in model.J for m in model.M)
    model.makespan = Constraint(model.J, model.M, rule=makespan_rule)
    
    # Objective: Minimize makespan
    model.obj = Objective(expr=model.Cmax, sense=minimize)
    
    # Solve
    solver = SolverFactory('highs')
    solver.solve(model)
    
    # Extract results
    schedule = []
    for j in model.J:
        for m in model.M:
            if model.x[j, m].value == 1:
                schedule.append((jobs[j-1], machines[m-1], processing_times[j-1][m-1]))
    
    return schedule, model.Cmax.value

# Streamlit GUI
st.title("Job Shop Scheduling Solver")

uploaded_file = st.file_uploader("Upload Job Shop Data (CSV)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:", df)
    
    jobs = df['Job'].tolist()
    machines = df.columns[1:].tolist()
    processing_times = df.iloc[:, 1:].values.tolist()
    
    if st.button("Solve Scheduling Problem"):
        schedule, makespan = solve_job_shop(jobs, machines, processing_times)
        
        st.write("### Optimized Schedule")
        for job, machine, time in schedule:
            st.write(f"**Job {job} -> Machine {machine}: {time} time units**")
        
        st.write(f"### Minimum Makespan: {makespan}")
