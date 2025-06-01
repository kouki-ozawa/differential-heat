import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from scipy.interpolate import interp1d

# 日本語フォントの設定
rcParams['font.family'] = 'MS Gothic'

# 入出力ファイルのパス
from dotenv import load_dotenv
import os

load_dotenv()

csv_path = os.getenv('CSV_PATH') #csvファイルのパス
output1 = os.getenv('OUTPUT1') #グラフ1のパス
output2 = os.getenv('OUTPUT2') #グラフ2のパス
output3 = os.getenv('OUTPUT3') #グラフ3のパス
output4 = os.getenv('OUTPUT4') #グラフ4のパス
output5 = os.getenv('OUTPUT5') #グラフ5のパス
output6 = os.getenv('OUTPUT6') #グラフ6のパス
output7 = os.getenv('OUTPUT7') #表1のパス



def create_temperature_plot(x, y, output_path):
    """温度のグラフを作成する関数"""
    plt.figure()
    
    # mV → K への変換
    y = np.array(y) / 1000.0  # μV → mV
    temperature_k = 24.184 * y + 0.9812
    
    plt.plot(x, temperature_k, marker='o', markersize=3, alpha=0.5)
    plt.xlabel("t[s]")
    plt.ylabel("T[K]")  # 単位をKに変更
    
    # メインの目盛りと補助目盛りを設定
    ax = plt.gca()
    
    # x軸の補助目盛り
    ax.xaxis.set_minor_locator(plt.MultipleLocator(500))   # 補助目盛り間隔
    
    # y軸の補助目盛り
    ax.yaxis.set_minor_locator(plt.MultipleLocator(5))    # 補助目盛り間隔
    
    # グリッド線の設定
    plt.grid(True, which='major', linestyle='-')
    plt.grid(True, which='minor', linestyle='--', alpha=0.3)
    plt.xlim(left=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=600)
    plt.close()

def create_delta_temperature_plot(x, y, output_path):
    """温度差のグラフを作成する関数"""
    plt.figure()
    
    # データの単位変換
    y = np.array(y) / 1000.0  # μV → mV
    y = y * 25.033  # mV → K
    
    # オリジナルのデータポイントをプロット
    plt.plot(x, y, 'b.', markersize=3, alpha=0.5)
    
    # 自然なスプライン補間
    f = interp1d(x, y, kind='cubic')  # 3次スプライン補間
    x_smooth = np.linspace(min(x), max(x), 300)
    y_smooth = f(x_smooth)
    
    # 補間したデータをプロット
    plt.plot(x_smooth, y_smooth, 'b-', linewidth=0.5)
    
    plt.xlabel("t[s]")
    plt.ylabel("ΔT[K]")  # 単位をKに変更
    
    
    # メインの目盛りと補助目盛りを設定
    ax = plt.gca()
    
    # x軸の補助目盛り
    ax.xaxis.set_minor_locator(plt.MultipleLocator(500))   # 補助目盛り間隔
    
    # y軸の補助目盛り
    ax.yaxis.set_minor_locator(plt.MultipleLocator(10))    # 補助目盛り間隔
    
    # グリッド線の設定
    plt.grid(True, which='major', linestyle='-')
    plt.grid(True, which='minor', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=800)
    plt.close()

def crommel_almer_thermocouple(x, y, output_path):
    """クロムメル-アルメル熱電対の温度を計算する関数"""
    # μV → mV への変換
    y_mv = y / 1000
    
    # 温度換算（℃）
    temperature = 24.184 * y_mv + 0.9812
    
    # グラフを作成
    plt.figure()
    plt.plot(y_mv, temperature, marker='o')
    plt.xlabel("熱起電力 [mV]")
    plt.ylabel("温度 [℃]")
    plt.title("クロムメル-アルメル熱電対 温度換算")
    plt.grid(True)
    
    # x軸の範囲を0から設定
    plt.xlim(left=0)
    
    # 目盛りの表示形式を設定（小数点4桁）
    ax = plt.gca()
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
   
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=800)
    plt.close()
    
    return temperature

def calculate_temperature_difference(x, y, output_path):
    """温度差の計算とグラフ作成"""

    y_mv = y / 1000
    delta_temperature = 25.033 * y_mv  # 係数を25.033に修正

    plt.figure()
    plt.plot(y_mv, delta_temperature, marker='o')  # x軸をy_mvに変更
    plt.xlabel("起電力 (mV)")
    plt.ylabel("温度 (℃)")
    plt.title("クロメル-アルメル熱電対")
    plt.grid(True)

    # 軸の範囲設定
    plt.xlim(0.000, 1.000)
    plt.ylim(0, 25)

    # 目盛りの表示形式を設定（x軸は小数点3桁）
    ax = plt.gca()
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=800)
    plt.close()

def draw_baseline(x, y, output_path):
    """DTAの基線を描画する関数(グラフ５)"""
    
    # NumPy配列に変換
    x = np.array(x)
    y = np.array(y) / 1000.0  # μV → mV
    y = y * 25.033  # mV → K
    
    plt.figure(figsize=(12, 8))
    
    # オリジナルのデータをプロット（点のみ）
    plt.plot(x, y, 'b.', markersize=5)
    
    # 安定領域を定義（配列のインデックスで指定）
    pre_transition = x < 1000    # 相転移前
    final_region = x > 1600      # 最終安定領域（1600秒以降）
    
    # 下り部分の領域を定義
    descent_region = (x >= 1250) & (x <= 1450)  # 下り部分の範囲を明確に指定
    
    # 安定領域のデータを抽出
    x_pre = x[pre_transition]
    y_pre = y[pre_transition]
    x_final = x[final_region]
    y_final = y[final_region]
    
    # 下り部分のデータを抽出
    x_descent = x[descent_region]
    y_descent = y[descent_region]
    
    if len(x_pre) > 0 and len(x_final) > 0:  # データが存在することを確認
        # 各領域の平均値を計算（安定領域用）
        x_pre_mean = np.mean(x_pre)
        y_pre_mean = np.mean(y_pre)
        x_final_mean = np.mean(x_final)
        y_final_mean = np.mean(y_final)
        
        # 2点を通る直線の傾きと切片を計算（安定領域用）
        slope1 = (y_final_mean - y_pre_mean) / (x_final_mean - x_pre_mean)
        intercept1 = y_pre_mean - slope1 * x_pre_mean
        
        # 指定したx座標（ta）での値を計算
        ta_x = 780  # ここで任意のx座標を指定
        # 最も近い実データ点を見つける
        idx = np.abs(x - ta_x).argmin()
        ta_x = x[idx]  # 実際のデータ点のx座標
        ta_y = y[idx]  # 実際のデータ点のy座標
        
        # ta点を矢印で表示（直線の矢印に変更）
        plt.annotate('t$_{a}$ (' + f'{ta_x:.1f}s)',
                    xy=(ta_x, ta_y), xycoords='data',
                    xytext=(ta_x, plt.ylim()[0]), textcoords='data',
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),  # rad=0で直線に
                    horizontalalignment='center', verticalalignment='top')
        
        # フィッティング結果を表示（安定領域用）
        print(f"基線1の方程式: y = {slope1:.6f}x + {intercept1:.6f}")
        print(f"t\u2090: ({ta_x:.2f}, {ta_y:.2f})")  
        
        # 基線1の描画（安定領域を結ぶ基線）（相転移前と相転移後の平均値を結ぶ直線）
        baseline_x1 = np.array([min(x), max(x)])
        baseline_y1 = slope1 * baseline_x1 + intercept1
        plt.plot(baseline_x1, baseline_y1, 'r--', linewidth=2,
                label=f'基線1 (y = {slope1:.6f}x + {intercept1:.6f})')
        
        # 平均値点を強調表示
        #plt.plot([x_pre_mean, x_final_mean], [y_pre_mean, y_final_mean], 'ro', markersize=8)
    
    if len(x_descent) > 0:  # 下り部分のデータが存在することを確認
        # 下り部分の直線フィッティング
        coeffs = np.polyfit(x_descent, y_descent, 1)
        slope2 = coeffs[0]
        intercept2 = coeffs[1]
        
        # フィッティング結果を表示（下り部分）
        print(f"基線2の方程式: y = {slope2:.6f}x + {intercept2:.6f}")
        
        # 2つの直線の交点を計算
        x_intersect = (intercept2 - intercept1) / (slope1 - slope2)
        y_intersect = slope1 * x_intersect + intercept1
        
        # 基線2の描画（全て破線で統一）
        x_points = np.array([1250, 1450, x_intersect])
        x_points.sort()  # 時間順に並べ替え
        baseline_x2 = x_points
        baseline_y2 = slope2 * baseline_x2 + intercept2
        plt.plot(baseline_x2, baseline_y2, 'g--', linewidth=2,
                label=f'基線2 (y = {slope2:.6f}x + {intercept2:.6f})')
        
        # 交点を表示
        plt.plot(x_intersect, y_intersect, 'ko', markersize=8,
                label=f'基線１と基線２の交点 ({x_intersect:.1f}, {y_intersect:.1f})')
        
        # 交点（te）を矢印で表示
        plt.annotate('t$_{e}$ (' + f'{x_intersect:.1f}s)',
                    xy=(x_intersect, y_intersect), xycoords='data',
                    xytext=(x_intersect, plt.ylim()[0]), textcoords='data',
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
                    horizontalalignment='center', verticalalignment='top')
        
        # 下り部分のデータ点を表示
        plt.plot(x_descent, y_descent, 'g.', alpha=0.3)
        
        print(f"t\u2091: ({x_intersect:.2f}, {y_intersect:.2f})")  # te
        
        # データの最小値（tp）を検出
        min_idx = np.argmin(y)
        tp_x = x[min_idx]
        tp_y = y[min_idx]
        
        # tp点を矢印で表示
        plt.annotate('t$_{p}$ (' + f'{tp_x:.1f}s)',
                    xy=(tp_x, tp_y), xycoords='data',
                    xytext=(tp_x, plt.ylim()[0]), textcoords='data',
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
                    horizontalalignment='center', verticalalignment='top')
        
        # x = 1500の垂直線を描画
        vertical_x = 1500
        plt.axvline(x=vertical_x, color='y', linestyle='--', linewidth=2,
                    label=f'基線3 (t$_b$ = {vertical_x}s)')
        
        # tb点をx軸上に表示
        plt.annotate('t$_b$',
                    xy=(vertical_x, plt.ylim()[0]), xycoords='data',
                    xytext=(vertical_x, plt.ylim()[0]), textcoords='data',
                    horizontalalignment='center', verticalalignment='top')
        
        print(f"t\u209A: ({tp_x:.2f}, {tp_y:.2f})")  # tp
    
    # グラフの設定
    plt.xlabel('t[s]')
    plt.ylabel('ΔT[K]')  # 単位をKに変更
    #plt.title('示差熱分析 (DTA) - 基線')
    plt.grid(True)
    plt.legend()
    
    # メインの目盛りと補助目盛りを設定
    ax = plt.gca()
    
    # x軸の補助目盛り
    ax.xaxis.set_minor_locator(plt.MultipleLocator(500))   # 補助目盛り間隔
    
    # y軸の補助目盛り
    ax.yaxis.set_minor_locator(plt.MultipleLocator(10))    # 補助目盛り間隔
    
    # グリッド線の設定
    plt.grid(True, which='major', linestyle='-')
    plt.grid(True, which='minor', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=800)
    plt.close()
    
    if len(x_pre) > 0 and len(x_final) > 0 and len(x_descent) > 0:
        return (slope1, intercept1), (slope2, intercept2), (x_intersect, y_intersect)
    else:
        return None, None, None

def draw_baseline_with_area(x, y, output_path):
    """DTAの基線を描画し、面積Aを塗りつぶす関数(グラフ６)"""
    # 基本的な処理は draw_baseline と同じ
    x = np.array(x) #x軸のデータ
    y = np.array(y) / 1000.0 #y軸のデータ
    y = y * 25.033 #y軸のデータ
    
    plt.figure(figsize=(12, 8))
    
    # オリジナルのデータをプロット
    plt.plot(x, y, 'b.', markersize=5) 
    
    # 安定領域を定義
    pre_transition = x < 1000 #相転移前
    final_region = x > 1600 #最終安定領域
    descent_region = (x >= 1250) & (x <= 1450) #下り部分
    
    # データを抽出
    x_pre = x[pre_transition] #相転移前
    y_pre = y[pre_transition] #相転移前
    x_final = x[final_region] #最終安定領域
    y_final = y[final_region] #最終安定領域
    x_descent = x[descent_region] #下り部分
    y_descent = y[descent_region] #下り部分
    
    if len(x_pre) > 0 and len(x_final) > 0:
        # 基線1の計算
        x_pre_mean = np.mean(x_pre) #安定領域の平均値
        y_pre_mean = np.mean(y_pre) #安定領域の平均値
        x_final_mean = np.mean(x_final) #安定領域の平均値
        y_final_mean = np.mean(y_final) #安定領域の平均値
        slope1 = (y_final_mean - y_pre_mean) / (x_final_mean - x_pre_mean) #安定領域の傾き
        intercept1 = y_pre_mean - slope1 * x_pre_mean
        
        # 面積Sの塗りつぶし用のデータ作成
        vertical_x = 1500
        ta_x = 780  # taの位置
        
        # taからx = 1500までのデータを抽出
        mask = (x >= ta_x) & (x <= vertical_x)
        x_fill = x[mask]
        y_fill = y[mask]
        
        # 基線1上の点を計算
        y_baseline = slope1 * x_fill + intercept1
        
        # 区分求積法で面積を計算
        # 各区間の幅（x座標の差分）
        dx = np.diff(x_fill)
        # 各区間の平均の高さ（基線からの差）
        y_diff = y_fill - y_baseline  # 基線からの差
        y_heights = (y_diff[:-1] + y_diff[1:]) / 2  # 台形の平均の高さ
        # 面積の計算（各区間の面積の合計）
        area = np.sum(dx * y_heights)
        print(f"面積S = {abs(area):.2f} K·s")       
        
        # 塗りつぶし
        plt.fill_between(x_fill, y_fill, y_baseline, color='gray', alpha=0.2, 
                        label=f'面積S = {abs(area):.2f} K·s')
        
        # ta点を矢印で表示
        idx = np.abs(x - ta_x).argmin()
        ta_x = x[idx]
        ta_y = y[idx]
        plt.annotate('ta',
                    xy=(ta_x, ta_y), xycoords='data',
                    xytext=(ta_x, plt.ylim()[0]), textcoords='data',
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
                    horizontalalignment='center', verticalalignment='top')
        plt.plot(ta_x, ta_y, 'ko', markersize=8, label=f'ta ({ta_x:.1f}s)')
        
        #print(f"t\u2090: ({ta_x:.2f}, {ta_y:.2f})")
        
        # 基線1の描画
        baseline_x1 = np.array([min(x), max(x)])
        baseline_y1 = slope1 * baseline_x1 + intercept1
        plt.plot(baseline_x1, baseline_y1, 'r--', linewidth=2,
                label=f'基線1 (y = {slope1:.6f}x + {intercept1:.6f})')
        
        # 平均値点を表示
        #plt.plot([x_pre_mean, x_final_mean], [y_pre_mean, y_final_mean], 'ro', markersize=8)
        
        if len(x_descent) > 0:
            # 基線2の計算と描画
            coeffs = np.polyfit(x_descent, y_descent, 1)
            slope2 = coeffs[0]
            intercept2 = coeffs[1]
            
            x_intersect = (intercept2 - intercept1) / (slope1 - slope2)
            y_intersect = slope1 * x_intersect + intercept1
            
            baseline_x2 = np.array([1250, 1450, x_intersect])
            baseline_x2.sort()
            baseline_y2 = slope2 * baseline_x2 + intercept2
            plt.plot(baseline_x2, baseline_y2, 'g--', linewidth=2,
                    label=f'基線2 (y = {slope2:.6f}x + {intercept2:.6f})')
            
            plt.plot(x_descent, y_descent, 'g.', alpha=0.3)
            
            # 交点を表示
            plt.plot(x_intersect, y_intersect, 'ko', markersize=8,
                    label=f'基線１と基線２の交点 ({x_intersect:.1f}, {y_intersect:.1f})')
            
            # te点を矢印で表示
            plt.plot(x_intersect, y_intersect, 'ko', markersize=8,
                    label=f't$_e$ ({x_intersect:.1f}s)')
            plt.annotate('t$_{e}$',
                        xy=(x_intersect, y_intersect), xycoords='data',
                        xytext=(x_intersect, plt.ylim()[0]), textcoords='data',
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
                        horizontalalignment='center', verticalalignment='top')
            
            # データの最小値（tp）を検出
            min_idx = np.argmin(y)
            tp_x = x[min_idx]
            tp_y = y[min_idx]
            
            # tp点を矢印で表示
            plt.plot(tp_x, tp_y, 'ko', markersize=8, label=f't$_p$ ({tp_x:.1f}s)')
            plt.annotate('t$_{p}$',
                        xy=(tp_x, tp_y), xycoords='data',
                        xytext=(tp_x, plt.ylim()[0]), textcoords='data',
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
                        horizontalalignment='center', verticalalignment='top')
    
    # x = 1500の垂直線を描画
    plt.axvline(x=vertical_x, color='y', linestyle='--', linewidth=2,
                label=f'基線3 (t$_b$ = {vertical_x}s)')
    
    # tb点をx軸上に表示
    plt.annotate('t$_b$',
                xy=(vertical_x, plt.ylim()[0]), xycoords='data',
                xytext=(vertical_x-20, plt.ylim()[0]), textcoords='data',
                horizontalalignment='center', verticalalignment='top')
    
    # グラフの設定
    plt.xlabel('t[s]')
    plt.ylabel('ΔT[K]')
    plt.title('示差熱分析 (DTA) - 基線と面積S')
    plt.grid(True)
    plt.legend()
    
    # メインの目盛りと補助目盛りを設定
    ax = plt.gca()
    ax.xaxis.set_minor_locator(plt.MultipleLocator(500))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(10))
    plt.grid(True, which='major', linestyle='-')
    plt.grid(True, which='minor', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=800)
    plt.close()
    return area

def calculate_heat_of_fusion(area):
    """
    融解熱と熱量を計算する関数

    """
    # ステアリン酸の熱量
    stearic_acid = 199.5 # [J/g] <-----教科書の定義
    #使用したステアリン酸の質量
    mass_stearic_acid = 0.1549# [g] <-----実験で測定した値

    H = stearic_acid * mass_stearic_acid

    #κの算出（面積の絶対値を使用）
    κ = H / abs(area)
    print(f"κ = {κ:.5f}[J/K・s]")

    #融解による熱量の計算
    heat = mass_stearic_acid * stearic_acid
    print(f"融解による熱量 = {heat:.2f}[J]")
   
    
    return κ, heat

def create_table():
    """表１を作成する関数"""
    # CSVファイルを読み込む
    df = pd.read_csv(csv_path, nrows=116)  # 116行目までのデータを読み込む
    
    # データを半分に分割
    half_length = len(df) // 2
    df1 = df.iloc[:half_length]  # 前半
    df2 = df.iloc[half_length:]  # 後半
    
    # 表のサイズとフォントサイズを設定
    fig = plt.figure(figsize=(16, 8))  # サイズを小さく調整
    
    # サブプロット間の間隔を調整
    gs = fig.add_gridspec(1, 2, wspace=0.1)  # wspaceで間隔を調整
    
    # 左側の表を作成（前半）
    ax1 = fig.add_subplot(gs[0])
    table1 = plt.table(
        cellText=df1.values,
        colLabels=df1.columns,
        loc='center',
        cellLoc='center',
        colColours=['#f2f2f2']*len(df1.columns),
        cellColours=[['#ffffff']*len(df1.columns)]*len(df1)
    )
    
    # 右側の表を作成（後半）
    ax2 = fig.add_subplot(gs[1])
    table2 = plt.table(
        cellText=df2.values,
        colLabels=df2.columns,
        loc='center',
        cellLoc='center',
        colColours=['#f2f2f2']*len(df2.columns),
        cellColours=[['#ffffff']*len(df2.columns)]*len(df2)
    )
    
    # 表のスタイル調整
    for table in [table1, table2]:
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        
        # セルの幅を内容に合わせて自動調整
        for col in range(len(df.columns)):
            # 各列の最大文字数を計算（ヘッダーも含める）
            max_length = max(
                len(str(cell.get_text().get_text())) 
                for cell in table._cells.values() 
                if cell.get_text()._x == table._cells[(0, col)].get_text()._x
            )
            # 文字数に応じて列幅を調整（係数は適宜調整）
            table._cells[(0, col)].set_width(max_length * 0.025)  # 係数を小さく
            for row in range(1, len(table._cells) // len(df.columns)):
                table._cells[(row, col)].set_width(max_length * 0.025)  # 係数を小さく
        
        # 高さは統一
        for cell in table._cells.values():
            cell.set_height(0.02)  # 高さも調整
    
    # 軸を非表示
    plt.axis('off')
    ax1.axis('off')
    ax2.axis('off')
    
    # 余白調整
    plt.tight_layout(pad=0.1)  # padを小さく設定
    
    # 保存（余白を最小限に）
    plt.savefig(output7, dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def main():
    # CSV読み込み（ヘッダーなし想定） — 1列目・2列目・3列目を読み込む
    df = pd.read_csv(csv_path, header=None, usecols=[0, 1, 2])
    
    # 数値変換→NaN行を一括削除
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    
    # グラフ1: 1列目 vs 2列目
    x1 = df.iloc[:, 0]
    y1 = df.iloc[:, 1]
    create_temperature_plot(x1, y1, output1)
    
    # グラフ2: 1列目 vs 3列目
    x2 = df.iloc[:, 0]
    y2 = df.iloc[:, 2]
    create_delta_temperature_plot(x2, y2, output2)
    
    # グラフ3: 熱電対の温度換算
    crommel_almer_thermocouple(x1, y1, output3)
    
    # グラフ4: 温度差の計算とグラフ作成
    calculate_temperature_difference(x1, y1, output4)
    
    # グラフ5: DTAの基線描画
    draw_baseline(x2, y2, output5)
    
    # グラフ6: 面積Sを塗りつぶしたグラフと面積の計算
    area = draw_baseline_with_area(x2, y2, output6)
    
    # 融解熱と熱量の計算
    κ, heat = calculate_heat_of_fusion(area)

    # 表１の作成
    create_table()

   


if __name__ == "__main__":
    main()
