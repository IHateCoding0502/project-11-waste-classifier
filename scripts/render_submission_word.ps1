param([Parameter(Mandatory=$true)][string]$InputDocx,[Parameter(Mandatory=$true)][string]$OutputPdf,[switch]$UpdateFields)
$ErrorActionPreference = 'Stop'
$taskPdfDirectory = [System.IO.Path]::GetDirectoryName([System.IO.Path]::GetFullPath($OutputPdf))
[System.IO.Directory]::CreateDirectory($taskPdfDirectory) | Out-Null
$taskWord = $null
$taskDocument = $null
try {
    $taskWord = New-Object -ComObject Word.Application
    $taskWord.Visible = $false
    $taskWord.DisplayAlerts = 0
    $taskDocument = $taskWord.Documents.Open([System.IO.Path]::GetFullPath($InputDocx), $false, (-not $UpdateFields.IsPresent))
    if ($UpdateFields) {
        $taskDocument.Fields.Update() | Out-Null
        foreach ($taskTOC in $taskDocument.TablesOfContents) { $taskTOC.Update() }
        $taskDocument.Repaginate()
        foreach ($taskTOC in $taskDocument.TablesOfContents) { $taskTOC.UpdatePageNumbers() }
        $taskDocument.Save()
    }
    $taskDocument.ExportAsFixedFormat([System.IO.Path]::GetFullPath($OutputPdf), 17)
    Write-Output ('Pages: ' + $taskDocument.ComputeStatistics(2))
} finally {
    if ($null -ne $taskDocument) { $taskDocument.Close(0) }
    if ($null -ne $taskWord) { $taskWord.Quit() }
}
