# Enter credentials in a private local window. They are encrypted by Windows.
# This script never places trades or changes your Alpaca account.
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
[System.Windows.Forms.Application]::EnableVisualStyles()

$form = New-Object System.Windows.Forms.Form
$form.Text = 'Connect Alpaca for ENEL 525'
$form.Size = New-Object System.Drawing.Size(610, 370)
$form.StartPosition = 'CenterScreen'
$form.FormBorderStyle = 'FixedDialog'
$form.MaximizeBox = $false
$form.Font = New-Object System.Drawing.Font('Segoe UI', 10)

$instructions = New-Object System.Windows.Forms.Label
$instructions.Location = New-Object System.Drawing.Point(20, 18)
$instructions.Size = New-Object System.Drawing.Size(550, 82)
$instructions.Text = "In Alpaca, switch to Paper Trading and open API Keys. Paste the key ID and secret below. They stay on this computer in an encrypted file. This project only downloads historical data."
$form.Controls.Add($instructions)

$keyLabel = New-Object System.Windows.Forms.Label
$keyLabel.Text = 'API key ID'
$keyLabel.Location = New-Object System.Drawing.Point(20, 112)
$keyLabel.AutoSize = $true
$form.Controls.Add($keyLabel)
$keyBox = New-Object System.Windows.Forms.TextBox
$keyBox.Location = New-Object System.Drawing.Point(150, 108)
$keyBox.Size = New-Object System.Drawing.Size(415, 28)
$keyBox.UseSystemPasswordChar = $true
$form.Controls.Add($keyBox)

$secretLabel = New-Object System.Windows.Forms.Label
$secretLabel.Text = 'Secret key'
$secretLabel.Location = New-Object System.Drawing.Point(20, 158)
$secretLabel.AutoSize = $true
$form.Controls.Add($secretLabel)
$secretBox = New-Object System.Windows.Forms.TextBox
$secretBox.Location = New-Object System.Drawing.Point(150, 154)
$secretBox.Size = New-Object System.Drawing.Size(415, 28)
$secretBox.UseSystemPasswordChar = $true
$form.Controls.Add($secretBox)

$status = New-Object System.Windows.Forms.Label
$status.Location = New-Object System.Drawing.Point(20, 204)
$status.Size = New-Object System.Drawing.Size(550, 45)
$status.Text = 'The keys will not be printed or sent to the chat.'
$form.Controls.Add($status)

$save = New-Object System.Windows.Forms.Button
$save.Text = 'Save keys locally'
$save.Location = New-Object System.Drawing.Point(360, 268)
$save.Size = New-Object System.Drawing.Size(205, 36)
$credentialPath = Join-Path $PSScriptRoot '.alpaca-credentials.xml'
$save.Add_Click({
    if ([string]::IsNullOrWhiteSpace($keyBox.Text) -or [string]::IsNullOrWhiteSpace($secretBox.Text)) {
        $status.Text = 'Please fill in both fields.'
        return
    }
    try {
        $secureKey = ConvertTo-SecureString $keyBox.Text.Trim() -AsPlainText -Force
        $secureSecret = ConvertTo-SecureString $secretBox.Text.Trim() -AsPlainText -Force
        [PSCustomObject]@{ApiKey=$secureKey; Secret=$secureSecret} | Export-Clixml -LiteralPath $credentialPath
        $keyBox.Clear()
        $secretBox.Clear()
        $form.DialogResult = [System.Windows.Forms.DialogResult]::OK
        $form.Close()
    } catch {
        $status.Text = 'Could not save the encrypted file. Check folder access and try again.'
    }
})
$form.Controls.Add($save)
$form.AcceptButton = $save
[void]$form.ShowDialog()
$form.Dispose()
