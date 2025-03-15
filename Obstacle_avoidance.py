import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from scipy.optimize import minimize
from matplotlib.patches import Polygon

def b_spline(t, control_points, degree=3, knots=None):
    """Generates a B-spline curve given control points."""
    control_points = np.asarray(control_points)
    n = len(control_points)
    
    if n < degree + 1:
        raise ValueError("Number of control points must be at least degree + 1")

    if knots is None:
        # Generate clamped knot vector
        start = [0.0] * (degree + 1)
        end = [1.0] * (degree + 1)
        num_internal = n - degree - 1
        internal = np.linspace(0, 1, num_internal + 2)[1:-1]
        knots = np.concatenate([start, internal, end])
    
    if len(knots) != n + degree + 1:
        # Adjust knot vector to correct length
        start = [0.0] * (degree + 1)
        end = [1.0] * (degree + 1)
        num_internal = n - degree - 1
        internal = np.linspace(0, 1, num_internal + 2)[1:-1]
        knots = np.concatenate([start, internal, end])

    spl_x = BSpline(knots, control_points[:, 0], degree)
    spl_y = BSpline(knots, control_points[:, 1], degree)
    return np.column_stack([spl_x(t), spl_y(t)])

def obstacle_avoidance(control_points, obstacles, degree=3, safety_margin=0.5):
    """Optimizes B-spline control points to avoid obstacles with fixed start/end."""
    n = len(control_points)
    # Precompute correct knot vector
    start = [0.0] * (degree + 1)
    end = [1.0] * (degree + 1)
    num_internal = n - degree - 1
    internal = np.linspace(0, 1, num_internal + 2)[1:-1]
    knots = np.concatenate([start, internal, end])
    
    # Store original start and end points
    original_start = control_points[0].copy()
    original_end = control_points[-1].copy()
    
    def objective_function(x):
        """Minimize the total curve length while keeping smoothness."""
        new_control_points = x.reshape(control_points.shape)
        t = np.linspace(0, 1, 100)
        curve = b_spline(t, new_control_points, degree, knots=knots)
        dx = np.diff(curve[:, 0])
        dy = np.diff(curve[:, 1])
        return np.sum(np.sqrt(dx**2 + dy**2))  # Total length

    def distance_constraint(x, obs_points):
        """Ensure minimum distance from curve to obstacle edges."""
        new_control_points = x.reshape(-1, 2)
        t = np.linspace(0, 1, 100)
        curve = b_spline(t, new_control_points, degree, knots=knots)
        
        # Calculate distance to all obstacle edge points
        distances = np.min(np.linalg.norm(curve[:, np.newaxis] - obs_points, axis=2), axis=1)
        return np.min(distances) - safety_margin

    # Add edge points to obstacles for better distance approximation
    enhanced_obstacles = []
    for obstacle in obstacles:
        # Sample points along polygon edges
        edge_points = []
        for i in range(len(obstacle)):
            p1 = obstacle[i]
            p2 = obstacle[(i+1)%len(obstacle)]
            edge_points.extend(np.linspace(p1, p2, 10))  # 10 points per edge
        enhanced_obstacles.append(np.vstack(edge_points))
    
    # Creating constraints for each enhanced obstacle
    constraints = [{'type': 'ineq', 'fun': lambda x, obs=obs: distance_constraint(x, obs)} 
                   for obs in enhanced_obstacles]

    # Adding constraints to fix start and end control points
    def start_constraint(x):
        return x[0:2] - original_start  # First control point (x,y)
    
    def end_constraint(x):
        return x[-2:] - original_end  # Last control point (x,y)
    
    constraints.extend([
        {'type': 'eq', 'fun': start_constraint},
        {'type': 'eq', 'fun': end_constraint}
    ])

    # Run optimization
    result = minimize(objective_function, 
                      control_points.flatten(), 
                      constraints=constraints, 
                      method='SLSQP',
                      options={'maxiter': 500, 'disp': True})

    if not result.success:
        print("Optimization failed:", result.message)
        return control_points

    return result.x.reshape(control_points.shape)

# Data with explicit start/end constraints
control_points = np.array([[0, 0], [1, 2], [2, 3], [3, 1], [3.75, 2.5], [5, 0]])

# Generate obstacles with enhanced edge points
obstacles = [
    np.array([[2, 1], [2.5, 1.5], [2.5, 0.5]]),  
    np.array([[3.5, 2], [4, 2.5], [4.5, 2]])  
]

# Generate correct knot vector
degree = 3
n = len(control_points)
start = [0.0] * (degree + 1)
end = [1.0] * (degree + 1)
num_internal = n - degree - 1
internal = np.linspace(0, 1, num_internal + 2)[1:-1]
knots = np.concatenate([start, internal, end])

# Optimize control points
optimized_control_points = obstacle_avoidance(control_points, obstacles)

# Generate B-spline curves
t = np.linspace(0, 1, 100)
initial_curve = b_spline(t, control_points, knots=knots)
optimized_curve = b_spline(t, optimized_control_points, knots=knots)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(initial_curve[:, 0], initial_curve[:, 1], 'g--', lw=2, label="Initial B-spline")
plt.plot(optimized_curve[:, 0], optimized_curve[:, 1], 'b-', lw=2, label="Optimized B-spline")

# Control points
plt.scatter(control_points[:, 0], control_points[:, 1], c='green', s=80, 
            marker='o', edgecolors='k', label='Initial Control Points')
plt.scatter(optimized_control_points[:, 0], optimized_control_points[:, 1], c='blue', s=80,
            marker='s', edgecolors='k', label='Optimized Control Points')

# Plot original obstacle outlines
for obstacle in obstacles:
    poly = Polygon(obstacle, closed=True, color='red', alpha=0.3, lw=2)
    plt.gca().add_patch(poly)

plt.legend(loc='upper right')
plt.title('B-spline Optimization with Obstacle Avoidance', fontsize=14)
plt.xlabel('X Position', fontsize=12)
plt.ylabel('Y Position', fontsize=12)
plt.grid(True)
plt.axis('equal')  # Ensure proper aspect ratio
plt.show()
