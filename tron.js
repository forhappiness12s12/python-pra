import TronWeb from 'tronweb';
import { randomBytes } from 'crypto';
import fs from 'fs';

// Function to generate a random private key
function generateRandomPrivateKey() {
    const privateKeyBuffer = randomBytes(32);
    return privateKeyBuffer.toString('hex');
}

// Initialize TronWeb with the full node and API node URLs
const tronWeb = new TronWeb({
    fullHost: 'https://api.trongrid.io', // Mainnet URL
});

async function getWalletBalance(privateKey) {
    try {
        // Get the address from the private key
        const address = tronWeb.address.fromPrivateKey(privateKey);
        console.log('Wallet Address:', address);

        // Get the TRX balance of the address
        const balanceInSun = await tronWeb.trx.getBalance(address);
        const balanceInTRX = tronWeb.fromSun(balanceInSun);
        console.log(`TRX Balance for ${address}: ${balanceInTRX} TRX`);

        // Get the account details
        const account = await tronWeb.trx.getAccount(address);
        console.log(`Account details for ${address}:`, account);

        // Get the balances of all TRC-10 tokens
        const trc10TokenBalances = account.assetV2 || [];
        trc10TokenBalances.forEach(token => {
            console.log(`TRC-10 Token: ${token.key}, Balance: ${token.value}`);
        });

        // Get the balances of all TRC-20 tokens
        const trc20TokenBalances = [];
        if (account.trc20) {
            for (const tokenAddress of account.trc20) {
                const contract = await tronWeb.contract(ABI_FRAGMENT, tokenAddress);
                const tokenBalance = await contract.methods.balanceOf(address).call();
                const tokenBalanceFormatted = tronWeb.fromSun(tokenBalance);
                const tokenInfo = await tronWeb.trx.getTokenByID(tokenAddress);

                trc20TokenBalances.push({
                    address: tokenAddress,
                    name: tokenInfo.name,
                    symbol: tokenInfo.symbol,
                    balance: tokenBalanceFormatted
                });

                console.log(`TRC-20 Token: ${tokenInfo.name}, Symbol: ${tokenInfo.symbol}, Balance: ${tokenBalanceFormatted}`);
            }
        }

        // Optional: Save the balances to a file
        const data = {
            address: address,
            TRXBalance: `${balanceInTRX} TRX`,
            TRC10TokenBalances: trc10TokenBalances,
            TRC20TokenBalances: trc20TokenBalances
        };

        // If balance is not zero, save the private key and balance
        if (balanceInTRX > 0) {
            saveKeyAndBalance(data);
        }
    } catch (error) {
        console.error('Error fetching wallet balance:', error);
    }
}
function saveKeyAndBalance(data) {
    fs.writeFileSync(`${data.address}_balances.json`, JSON.stringify(data, null, 2), (err) => {
        if (err) throw err;
        console.log('Wallet balances saved!');
    });
}

async function generateKeysAndGetBalancesIndefinitely() {
    while (true) {
        const privateKey = generateRandomPrivateKey();
        console.log('Generated Private Key:', privateKey);
        await getWalletBalance(privateKey);
    }
}

generateKeysAndGetBalancesIndefinitely();
