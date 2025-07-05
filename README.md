## PADDY
<table>
  <tr>
    <td><img src="paddy.png" alt="paddy" width="500"></td>
    <td>
      <strong>PADDY</strong> is a deep learning model library focused on biological applications. It provides modular, reusable implementations to accelerate AI-driven biological discoveries.
    </td>
  </tr>
</table>

## Install
```bash
git clone https://github.com/GooLey1025/paddy.git
cd paddy
echo "export PATH=\$PATH:$(pwd)/src/paddy/scripts" >> ~/.bashrc
source ~/.bashrc
conda create -n paddy python=3.10
conda activate paddy
pip install -e .