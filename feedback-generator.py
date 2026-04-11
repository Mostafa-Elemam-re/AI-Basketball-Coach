import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os

class shootingAnalyzer:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        
        # Clean column names (strip whitespace) to prevent matching errors
        self.df.columns = self.df.columns.str.strip()
        
        # Check if Player_ID exists, if not, create a dummy one for a single-player file
        if 'Player_ID' not in self.df.columns:
            print("Notice: 'Player_ID' column not found. Treating file as single-player data.")
            self.df['Player_ID'] = 0

        # Proficient Benchmarks based on Cabarkapa et al. (2023)
        # Values represent mean peak angles at the "Set Point" or "Deepest Flexion"
        self.benchmarks = {
            'R_KNEE': {
                'mean': 113.1, 'std': 10.5, 'label': 'Knee Flexion',
                'too_low': "Knee bend is too deep. This can lead to slow release and loss of power consistency.",
                'too_high': "Standing too tall. You need to bend your knees more to generate power from the ground."
            },
            'R_ELBOW': {
                'mean': 65.4, 'std': 12.2, 'label': 'Elbow Flexion',
                'too_low': "Elbow is over-flexed. Ensure the ball isn't tucked too far behind your head.",
                'too_high': "Elbow is too 'flat'. Bring the ball closer to your face as you are releasing the ball to create a better angle."
            },
            'R_SHOULDER': {
                'mean': 102.7, 'std': 13.1, 'label': 'Shoulder Flexion',
                'too_low': "Shoulder/Elbow height is too low. Aim to keep the upper arm finish high to have a higher arc.",
                'too_high': "Shoulder angle is very high. Lower your shoulder at the release to not lose power on your shot."
            }
        }

    def analyze_player(self, player_id):
        player_data = self.df[self.df['Player_ID'] == player_id].copy()
        if player_data.empty:
            return None

        # 1. Identify "Set Point" (Moment of maximum elbow flexion during a shot)
        if 'R_ELBOW' not in player_data.columns:
            return None

        elbow_angles = player_data['R_ELBOW'].values
        elbow_series = pd.Series(elbow_angles).interpolate()
        if elbow_series.isnull().all():
            return None
            
        elbow_smooth = elbow_series.rolling(window=5, min_periods=1).mean().values
        set_point_idx = np.nanargmin(elbow_smooth)
        set_point_time = player_data.iloc[set_point_idx]['Timestamp']
        
        # 2. Extract Metrics
        results = {'metrics': {}, 'feedback': []}
        for joint in ['R_KNEE', 'R_ELBOW', 'R_SHOULDER']:
            if joint in player_data.columns:
                val = player_data.iloc[set_point_idx][joint]
                results['metrics'][joint] = val
                
                # Generate specific directional feedback
                target = self.benchmarks[joint]['mean']
                std = self.benchmarks[joint]['std']
                if val < (target - std):
                    results['feedback'].append(self.benchmarks[joint]['too_low'])
                elif val > (target + std):
                    results['feedback'].append(self.benchmarks[joint]['too_high'])
            else:
                results['metrics'][joint] = np.nan

        # 3. Kinematic Sequence Analysis (Templin et al. 2024)
        results['sequence_score'], seq_feedback = self._analyze_sequence(player_data)
        if seq_feedback:
            results['feedback'].append(seq_feedback)
        
        return results, set_point_time

    def _analyze_sequence(self, data):
        try:
            if 'R_KNEE' not in data.columns or 'R_ELBOW' not in data.columns:
                return "Missing Data", None
                
            knee_vel = np.gradient(data['R_KNEE'].interpolate().fillna(method='ffill').fillna(method='bfill').values)
            elbow_vel = np.gradient(data['R_ELBOW'].interpolate().fillna(method='ffill').fillna(method='bfill').values)
            
            peak_knee = np.argmax(knee_vel)
            peak_elbow = np.argmax(elbow_vel)
            
            if peak_elbow > peak_knee:
                return "Optimal", None
            else:
                return "Sub-optimal", "Sequence Issue: You are extending your arm before your legs finish driving. Aim for 'Legs then Arms'."
        except:
            return "Inconclusive", None

    def generate_report(self, player_id):
        analysis = self.analyze_player(player_id)
        if not analysis: return

        data, _ = analysis
        metrics = data['metrics']
        
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.set_axis_off()
        fig.patch.set_facecolor('#f4f4f4')
        
        # Header
        plt.text(0.5, 0.95, f"PLAYER {player_id}: SHOOTING BIOMECHANICS REPORT", fontsize=16, weight='bold', ha='center', color='#1a1a1a')
        plt.text(0.5, 0.91, f"Comparative Analysis vs. Proficient Shooting Standards", fontsize=10, ha='center', color='#555555')

        # Joint Data Table
        y_pos = 0.78
        plt.text(0.05, y_pos + 0.04, "METRIC ANALYSIS", fontsize=12, weight='bold', color='#2c3e50')
        
        for joint, val in metrics.items():
            if pd.isna(val) or joint not in self.benchmarks: continue
            
            target = self.benchmarks[joint]['mean']
            std = self.benchmarks[joint]['std']
            status = "PROFICIENT" if abs(val - target) < std else "ADJUSTMENT NEEDED"
            color = '#27ae60' if status == "PROFICIENT" else '#e74c3c'
            
            # Row Background
            rect = plt.Rectangle((0.04, y_pos-0.015), 0.92, 0.05, color='white', alpha=0.8, transform=ax.transAxes, zorder=-1)
            ax.add_patch(rect)
            
            plt.text(0.06, y_pos, self.benchmarks[joint]['label'], fontsize=11)
            plt.text(0.35, y_pos, f"Actual: {int(val)}°", fontsize=11)
            plt.text(0.55, y_pos, f"Standard: {int(target)}° (±{int(std)})", fontsize=11, color='#7f8c8d')
            plt.text(0.78, y_pos, status, fontsize=10, weight='bold', color=color)
            y_pos -= 0.07

        # Corrective Feedback Section
        y_pos -= 0.05
        plt.text(0.05, y_pos, "CORRECTIVE FEEDBACK & CUES", fontsize=12, weight='bold', color='#2c3e50')
        y_pos -= 0.05
        
        if not data['feedback']:
            plt.text(0.06, y_pos, "✓ All mechanics align with proficient shooting profiles. Maintain consistency.", fontsize=11, color='#27ae60')
        else:
            for fb in data['feedback']:
                plt.text(0.06, y_pos, f"• {fb}", fontsize=10.5, wrap=True)
                y_pos -= 0.05

        # Kinematic Summary
        plt.text(0.05, 0.22, "KINEMATIC SEQUENCE", fontsize=12, weight='bold', color='#2c3e50')
        seq_color = "#0db051" if data['sequence_score'] == "Optimal" else '#f39c12'
        plt.text(0.06, 0.17, f"Timing Profile: {data['sequence_score']}", fontsize=11, weight='bold', color=seq_color)
        plt.text(0.06, 0.13, "Optimal shooters maximize energy transfer by ensuring the leg drive peaks before the elbow extension.", fontsize=9, color='#7f8c8d')

        report_file = f"player_{player_id}_feedback_report.png"
        plt.savefig(report_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Enhanced feedback report saved: {report_file}")

if __name__ == "__main__":
    csv_path = "One Shot 26x.csv"
    if os.path.exists(csv_path):
        analyzer = shootingAnalyzer(csv_path)
        for p_id in analyzer.df['Player_ID'].unique():
            analyzer.generate_report(p_id)
    else:
        print(f"Error: {csv_path} not found.")